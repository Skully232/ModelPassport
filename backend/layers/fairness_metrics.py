"""
backend/layers/fairness_metrics.py

FairnessMetricsEngine — ModelPassport Pre-Deployment Fairness Auditor

Computes four core fairness dimensions for a trained sklearn-compatible model:
  1. Demographic Parity Difference    (via fairlearn)
  2. Equalized Odds Difference        (via fairlearn)
  3. Disparate Impact Ratio           (min-positive-rate / max-positive-rate)
  4. Individual Fairness Score        (Lipschitz-inspired consistency check)

Returns a structured result dict containing:
  - per_attribute_scores : detailed metrics for every protected attribute
  - overall_fairness_score: normalised 0–100 composite score
  - warnings             : human-readable list of detected violations
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# fairlearn public metric API (stable since 0.7)
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold constants (configurable via constructor kwargs)
# ---------------------------------------------------------------------------
_DPD_WARN_THRESHOLD = 0.1      # |demographic parity difference| > this → warn
_EOD_WARN_THRESHOLD = 0.1      # |equalized odds difference|     > this → warn
_DIR_WARN_THRESHOLD = 0.8      # disparate impact ratio < this   → warn
_IFS_WARN_THRESHOLD = 0.70     # individual fairness score < this → warn
_CRITICAL_MULTIPLIER = 1.5     # crossing threshold × multiplier → critical warning


class FairnessMetricsEngine:
    """
    Computes a comprehensive fairness audit for a trained sklearn model.

    Parameters
    ----------
    model : Any
        A trained, sklearn-compatible model exposing ``.predict()``
        (and optionally ``.predict_proba()``).
    dataset : pd.DataFrame
        The *test* split containing both feature columns and the target column.
    protected_attributes : List[str]
        Column names representing legally or ethically sensitive attributes
        (e.g. ``["gender", "race", "age_group"]``).
    target_column : str
        Name of the ground-truth label column inside ``dataset``.
    positive_outcome : Any, optional
        The label value considered a favourable outcome (default ``1``).
    dpd_threshold : float, optional
        Warning threshold for demographic parity difference (default 0.10).
    eod_threshold : float, optional
        Warning threshold for equalized odds difference (default 0.10).
    dir_threshold : float, optional
        Minimum acceptable disparate impact ratio (default 0.80).
    ifs_threshold : float, optional
        Minimum acceptable individual fairness score (default 0.70).
    individual_fairness_k : int, optional
        Number of nearest neighbours used in individual fairness check (default 5).
    """

    def __init__(
        self,
        model: Any,
        dataset: pd.DataFrame,
        protected_attributes: List[str],
        target_column: str,
        positive_outcome: Any = 1,
        *,
        dpd_threshold: float = _DPD_WARN_THRESHOLD,
        eod_threshold: float = _EOD_WARN_THRESHOLD,
        dir_threshold: float = _DIR_WARN_THRESHOLD,
        ifs_threshold: float = _IFS_WARN_THRESHOLD,
        individual_fairness_k: int = 5,
    ) -> None:
        if target_column not in dataset.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {list(dataset.columns)}"
            )

        missing_attrs = [a for a in protected_attributes if a not in dataset.columns]
        if missing_attrs:
            raise ValueError(
                f"Protected attributes not found in dataset: {missing_attrs}"
            )

        self.model = model
        self.dataset = dataset.copy()
        self.protected_attributes = protected_attributes
        self.target_column = target_column
        self.positive_outcome = positive_outcome

        # Thresholds
        self.dpd_threshold = dpd_threshold
        self.eod_threshold = eod_threshold
        self.dir_threshold = dir_threshold
        self.ifs_threshold = ifs_threshold
        self.individual_fairness_k = individual_fairness_k

        # Separate features / labels
        self.y_true: pd.Series = self.dataset[target_column]
        self.X: pd.DataFrame = self.dataset.drop(columns=[target_column])
        self.y_pred: np.ndarray = self._safe_predict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_predict(self) -> np.ndarray:
        """Run model inference, suppressing sklearn version warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(self.X)

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        """Convert numpy scalars to native Python types for JSON safety."""
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _encode_sensitive_feature(self, attr: str) -> pd.Series:
        """
        Return integer-encoded sensitive feature series.
        fairlearn accepts both raw categorical and numeric series;
        encoding ensures consistency across all attribute types.
        """
        col = self.dataset[attr]
        if pd.api.types.is_numeric_dtype(col):
            return col.astype(int)
        le = LabelEncoder()
        return pd.Series(le.fit_transform(col.astype(str)), index=col.index)

    def _positive_rate_by_group(self, attr: str) -> Dict[str, float]:
        """Return predicted positive rate per unique value of *attr*."""
        col = self.dataset[attr]
        rates: Dict[str, float] = {}
        for val in col.unique():
            mask = col == val
            preds_group = self.y_pred[mask]
            rates[str(val)] = float(np.mean(preds_group == self.positive_outcome))
        return rates

    def _true_positive_rate_by_group(
        self, attr: str
    ) -> Dict[str, float]:
        """Return true positive rate (recall) per unique group."""
        col = self.dataset[attr]
        y_true_arr = self.y_true.values
        tprs: Dict[str, float] = {}
        for val in col.unique():
            mask = (col == val).values
            y_t = y_true_arr[mask]
            y_p = self.y_pred[mask]
            pos_mask = y_t == self.positive_outcome
            if pos_mask.sum() == 0:
                tprs[str(val)] = 0.0
            else:
                tprs[str(val)] = float(np.mean(y_p[pos_mask] == self.positive_outcome))
        return tprs

    # ------------------------------------------------------------------
    # Metric 1 — Demographic Parity Difference
    # ------------------------------------------------------------------

    def _compute_demographic_parity(self, attr: str) -> Dict[str, Any]:
        """
        Compute demographic parity difference using fairlearn.

        DPD = max(P(Ŷ=1|A=a)) − min(P(Ŷ=1|A=a))  across groups *a*.
        Values closer to 0 indicate better parity.
        """
        sensitive = self._encode_sensitive_feature(attr)
        dpd = demographic_parity_difference(
            y_true=self.y_true,
            y_pred=self.y_pred,
            sensitive_features=sensitive,
        )
        positive_rates = self._positive_rate_by_group(attr)
        return {
            "demographic_parity_difference": self._to_serializable(dpd),
            "positive_rates_by_group": positive_rates,
        }

    # ------------------------------------------------------------------
    # Metric 2 — Equalized Odds Difference
    # ------------------------------------------------------------------

    def _compute_equalized_odds(self, attr: str) -> Dict[str, Any]:
        """
        Compute equalized odds difference using fairlearn.

        EOD = max over {TPR, FPR} of the demographic parity difference
        of that rate across groups. Closer to 0 is better.
        """
        sensitive = self._encode_sensitive_feature(attr)
        eod = equalized_odds_difference(
            y_true=self.y_true,
            y_pred=self.y_pred,
            sensitive_features=sensitive,
        )
        tpr_by_group = self._true_positive_rate_by_group(attr)
        return {
            "equalized_odds_difference": self._to_serializable(eod),
            "true_positive_rates_by_group": tpr_by_group,
        }

    # ------------------------------------------------------------------
    # Metric 3 — Disparate Impact Ratio
    # ------------------------------------------------------------------

    def _compute_disparate_impact(self, attr: str) -> Dict[str, Any]:
        """
        Disparate Impact Ratio = min_group_positive_rate / max_group_positive_rate.

        The EEOC 80% (Four-Fifths) rule flags ratios below 0.80 as adverse impact.
        A ratio of 1.0 indicates perfect group parity.
        """
        positive_rates = self._positive_rate_by_group(attr)
        rates_list = list(positive_rates.values())

        if not rates_list or max(rates_list) == 0:
            return {
                "disparate_impact_ratio": 0.0,
                "positive_rates_by_group": positive_rates,
                "privileged_group": None,
                "unprivileged_group": None,
                "pairwise_ratios": {},
            }

        max_rate = max(rates_list)
        min_rate = min(rates_list)
        di_ratio = min_rate / max_rate

        # Identify privileged (highest +rate) and unprivileged (lowest +rate) groups
        privileged = max(positive_rates, key=positive_rates.get)  # type: ignore[arg-type]
        unprivileged = min(positive_rates, key=positive_rates.get)  # type: ignore[arg-type]

        # Pairwise DI ratios vs. privileged group
        priv_rate = positive_rates[privileged]
        pairwise: Dict[str, float] = {}
        for grp, rate in positive_rates.items():
            if grp == privileged:
                continue
            pairwise[f"{grp}_vs_{privileged}"] = (
                float(rate / priv_rate) if priv_rate > 0 else 0.0
            )

        return {
            "disparate_impact_ratio": self._to_serializable(di_ratio),
            "positive_rates_by_group": positive_rates,
            "privileged_group": privileged,
            "unprivileged_group": unprivileged,
            "pairwise_ratios": pairwise,
        }

    # ------------------------------------------------------------------
    # Metric 4 — Individual Fairness Score
    # ------------------------------------------------------------------

    def _compute_individual_fairness(self) -> Dict[str, Any]:
        """
        Lipschitz-inspired individual fairness consistency check.

        Intuition: similar individuals should receive similar predictions.
        For each sample we find its *k* nearest neighbours in *feature space*
        (excluding protected attributes) and measure what fraction of those
        neighbours received the same prediction.  The score is the mean
        consistency across all samples.

        Score range: 0.0 (fully inconsistent) → 1.0 (perfectly consistent).
        """
        # Use only non-protected, numeric features for neighbour distance
        feature_cols = [
            c for c in self.X.columns
            if c not in self.protected_attributes
            and pd.api.types.is_numeric_dtype(self.X[c])
        ]

        if not feature_cols:
            logger.warning(
                "No numeric non-protected features found. "
                "Skipping individual fairness with score=None."
            )
            return {
                "individual_fairness_score": None,
                "note": (
                    "Skipped — no numeric non-protected features available "
                    "to compute neighbourhood similarity."
                ),
            }

        X_feats = self.X[feature_cols].copy()

        # Impute any remaining NaNs with column means for distance computation
        X_feats = X_feats.fillna(X_feats.mean())

        k = min(self.individual_fairness_k, len(X_feats) - 1)
        if k < 1:
            return {
                "individual_fairness_score": 1.0,
                "note": "Dataset too small for neighbour search; defaulting to 1.0.",
            }

        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        nbrs.fit(X_feats.values)
        indices = nbrs.kneighbors(X_feats.values, return_distance=False)

        consistencies: List[float] = []
        for i, neighbours in enumerate(indices):
            # Exclude self (index 0 is always the point itself)
            neighbour_idx = [n for n in neighbours if n != i][:k]
            if not neighbour_idx:
                consistencies.append(1.0)
                continue
            own_pred = self.y_pred[i]
            neighbour_preds = self.y_pred[neighbour_idx]
            consistencies.append(float(np.mean(neighbour_preds == own_pred)))

        ifs = float(np.mean(consistencies))
        return {
            "individual_fairness_score": ifs,
            "k_neighbours": k,
            "features_used": feature_cols,
        }

    # ------------------------------------------------------------------
    # Overall fairness score
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    def _aggregate_overall_score(
        self,
        per_attr: Dict[str, Dict[str, Any]],
        individual_fairness: Dict[str, Any],
    ) -> float:
        """
        Compute a composite fairness score in [0, 100].

        Weighting philosophy:
          - Demographic parity (25 pts per attribute, averaged)
          - Equalized odds    (25 pts per attribute, averaged)
          - Disparate impact  (25 pts per attribute, averaged)
          - Individual fairness (25 pts global)

        Each sub-score is normalised so that 0 = worst, 1 = best.
        """
        dpd_scores: List[float] = []
        eod_scores: List[float] = []
        dir_scores: List[float] = []

        for attr_data in per_attr.values():
            # DPD: abs difference; perfect=0, cap penalty at threshold×2
            dpd_val = abs(attr_data["demographic_parity"]["demographic_parity_difference"])
            dpd_scores.append(self._clamp(1.0 - dpd_val / (self.dpd_threshold * 2)))

            # EOD: same normalisation as DPD
            eod_val = abs(attr_data["equalized_odds"]["equalized_odds_difference"])
            eod_scores.append(self._clamp(1.0 - eod_val / (self.eod_threshold * 2)))

            # DIR: ideal=1.0, minimum=0; normalise so 1.0 → 1.0 score
            dir_val = attr_data["disparate_impact"]["disparate_impact_ratio"]
            dir_scores.append(self._clamp(dir_val))

        avg_dpd = float(np.mean(dpd_scores)) if dpd_scores else 1.0
        avg_eod = float(np.mean(eod_scores)) if eod_scores else 1.0
        avg_dir = float(np.mean(dir_scores)) if dir_scores else 1.0

        ifs_val = individual_fairness.get("individual_fairness_score")
        avg_ifs = self._clamp(float(ifs_val)) if ifs_val is not None else 1.0

        composite = (avg_dpd + avg_eod + avg_dir + avg_ifs) / 4.0
        return round(composite * 100, 2)

    # ------------------------------------------------------------------
    # Warning generation
    # ------------------------------------------------------------------

    def _generate_warnings(
        self,
        per_attr: Dict[str, Dict[str, Any]],
        individual_fairness: Dict[str, Any],
    ) -> List[str]:
        """Produce human-readable warning strings for every threshold breach."""
        warnings_list: List[str] = []

        for attr, data in per_attr.items():
            # — Demographic Parity —
            dpd_val = data["demographic_parity"]["demographic_parity_difference"]
            abs_dpd = abs(dpd_val)
            if abs_dpd > self.dpd_threshold * _CRITICAL_MULTIPLIER:
                warnings_list.append(
                    f"[CRITICAL] Demographic Parity on '{attr}': difference is "
                    f"{dpd_val:+.4f} (threshold ±{self.dpd_threshold:.2f}). "
                    f"Predicted positive rates diverge significantly across groups."
                )
            elif abs_dpd > self.dpd_threshold:
                warnings_list.append(
                    f"[WARNING] Demographic Parity on '{attr}': difference is "
                    f"{dpd_val:+.4f} (threshold ±{self.dpd_threshold:.2f}). "
                    f"Model may favour certain demographic groups in selection rate."
                )

            # — Equalized Odds —
            eod_val = data["equalized_odds"]["equalized_odds_difference"]
            abs_eod = abs(eod_val)
            if abs_eod > self.eod_threshold * _CRITICAL_MULTIPLIER:
                warnings_list.append(
                    f"[CRITICAL] Equalized Odds on '{attr}': difference is "
                    f"{eod_val:+.4f} (threshold ±{self.eod_threshold:.2f}). "
                    f"True positive rates vary substantially across protected groups."
                )
            elif abs_eod > self.eod_threshold:
                warnings_list.append(
                    f"[WARNING] Equalized Odds on '{attr}': difference is "
                    f"{eod_val:+.4f} (threshold ±{self.eod_threshold:.2f}). "
                    f"Model error rates are inconsistent across demographic groups."
                )

            # — Disparate Impact —
            dir_val = data["disparate_impact"]["disparate_impact_ratio"]
            priv = data["disparate_impact"].get("privileged_group", "N/A")
            unpriv = data["disparate_impact"].get("unprivileged_group", "N/A")
            if dir_val < self.dir_threshold * (2 - _CRITICAL_MULTIPLIER / _CRITICAL_MULTIPLIER):
                # dir < ~0.53 → critical (below threshold by >1.5×)
                pass  # handled below via unified check
            if dir_val == 0.0:
                warnings_list.append(
                    f"[CRITICAL] Disparate Impact on '{attr}': ratio is 0.0. "
                    f"The unprivileged group ('{unpriv}') receives zero positive outcomes."
                )
            elif dir_val < self.dir_threshold / _CRITICAL_MULTIPLIER:
                warnings_list.append(
                    f"[CRITICAL] Disparate Impact on '{attr}': ratio is {dir_val:.4f} "
                    f"(severe violation of the 80% Four-Fifths Rule). "
                    f"Privileged: '{priv}', Unprivileged: '{unpriv}'."
                )
            elif dir_val < self.dir_threshold:
                warnings_list.append(
                    f"[WARNING] Disparate Impact on '{attr}': ratio is {dir_val:.4f} "
                    f"(below the 80% Four-Fifths Rule threshold of {self.dir_threshold:.2f}). "
                    f"Privileged: '{priv}', Unprivileged: '{unpriv}'."
                )

        # — Individual Fairness —
        ifs_val = individual_fairness.get("individual_fairness_score")
        if ifs_val is not None:
            if ifs_val < self.ifs_threshold / _CRITICAL_MULTIPLIER:
                warnings_list.append(
                    f"[CRITICAL] Individual Fairness score is {ifs_val:.4f} "
                    f"(threshold ≥{self.ifs_threshold:.2f}). Similar individuals "
                    f"receive very different predictions — strong inconsistency detected."
                )
            elif ifs_val < self.ifs_threshold:
                warnings_list.append(
                    f"[WARNING] Individual Fairness score is {ifs_val:.4f} "
                    f"(threshold ≥{self.ifs_threshold:.2f}). Similar individuals "
                    f"do not consistently receive the same prediction."
                )

        return warnings_list

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute the full fairness audit pipeline.

        Returns
        -------
        dict with keys:
            status                  : "Pass" | "Fail"
            overall_fairness_score  : float in [0, 100]  (higher = fairer)
            per_attribute_scores    : dict keyed by protected attribute name
            individual_fairness     : dict with IFS score and metadata
            warnings                : List[str] — human-readable violation messages
            metrics_summary         : high-level per-attribute scalar digest
        """
        logger.info(
            "FairnessMetricsEngine: auditing %d samples, %d protected attributes.",
            len(self.dataset),
            len(self.protected_attributes),
        )

        per_attribute_scores: Dict[str, Any] = {}

        for attr in self.protected_attributes:
            logger.debug("Computing fairness metrics for attribute: '%s'", attr)
            try:
                dp_result = self._compute_demographic_parity(attr)
                eo_result = self._compute_equalized_odds(attr)
                di_result = self._compute_disparate_impact(attr)

                unique_vals = self.dataset[attr].nunique()
                per_attribute_scores[attr] = {
                    "unique_group_count": int(unique_vals),
                    "demographic_parity": dp_result,
                    "equalized_odds": eo_result,
                    "disparate_impact": di_result,
                }

            except Exception as exc:
                logger.error(
                    "Failed to compute metrics for attribute '%s': %s", attr, exc
                )
                per_attribute_scores[attr] = {
                    "error": str(exc),
                    "demographic_parity": {},
                    "equalized_odds": {},
                    "disparate_impact": {},
                }

        # Individual fairness (cross-attribute, single global score)
        try:
            individual_fairness = self._compute_individual_fairness()
        except Exception as exc:
            logger.error("Individual fairness computation failed: %s", exc)
            individual_fairness = {
                "individual_fairness_score": None,
                "error": str(exc),
            }

        # Composite score
        overall_score = self._aggregate_overall_score(
            per_attribute_scores, individual_fairness
        )

        # Warnings
        warnings_list = self._generate_warnings(per_attribute_scores, individual_fairness)

        # Compact summary (one scalar per metric per attribute for quick dashboard display)
        metrics_summary: Dict[str, Dict[str, Optional[float]]] = {}
        for attr, data in per_attribute_scores.items():
            if "error" in data:
                metrics_summary[attr] = {"error": data["error"]}
                continue
            metrics_summary[attr] = {
                "demographic_parity_difference": data["demographic_parity"].get(
                    "demographic_parity_difference"
                ),
                "equalized_odds_difference": data["equalized_odds"].get(
                    "equalized_odds_difference"
                ),
                "disparate_impact_ratio": data["disparate_impact"].get(
                    "disparate_impact_ratio"
                ),
            }

        result = {
            "status": "Fail" if warnings_list else "Pass",
            "overall_fairness_score": overall_score,
            "per_attribute_scores": per_attribute_scores,
            "individual_fairness": individual_fairness,
            "metrics_summary": metrics_summary,
            "warnings": warnings_list,
        }

        logger.info(
            "FairnessMetricsEngine complete — score: %.1f, status: %s, "
            "warnings: %d",
            overall_score,
            result["status"],
            len(warnings_list),
        )
        return result
