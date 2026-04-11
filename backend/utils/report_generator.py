"""
backend/utils/report_generator.py

CertificateGenerator — ModelPassport Certificate Issuance & Verification

Responsibilities:
  - Issue sequential, human-readable certificate IDs (MP-2026-000001 …)
  - Calculate a composite overall_score from all three quantitative audit layers
  - Map the score to a certification status tier
  - Produce a SHA-256 integrity hash of the full result payload
  - Persist every certificate to ``certificates_store.json``
  - Support point-in-time certificate verification via ``verify()``

All file I/O is protected by a ``threading.Lock`` so the class is safe for
use inside multi-threaded ASGI servers (e.g. uvicorn with multiple workers
sharing the same process).

Persistence files (relative to this file's parent directory, i.e. ``backend/``):
  - ``certificate_counter.json``  — monotonic counter, seeded at 0
  - ``certificates_store.json``   — dict keyed by certificate_id
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — both files live in ``backend/`` (one level above ``backend/utils/``)
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COUNTER_FILE = os.path.join(_BACKEND_DIR, "certificate_counter.json")
_STORE_FILE = os.path.join(_BACKEND_DIR, "certificates_store.json")

# ---------------------------------------------------------------------------
# Certification score thresholds
# ---------------------------------------------------------------------------
_CERTIFIED_THRESHOLD = 80.0
_CONDITIONAL_THRESHOLD = 60.0

# Year embedded in certificate IDs
_CERT_YEAR = "2026"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json_file(path: str, default: Any) -> Any:
    """Load a JSON file, returning ``default`` if the file does not exist."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read '%s': %s — using default.", path, exc)
        return default


def _write_json_file(path: str, data: Any) -> None:
    """Atomically write data to a JSON file (write-then-replace pattern)."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.error("Failed to write '%s': %s", path, exc)
        raise


def _compute_sha256(payload: Dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of ``json.dumps(payload, sort_keys=True)``."""
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _compute_overall_score(
    data_forensics: Dict[str, Any],
    stress_test: Dict[str, Any],
    fairness_metrics: Dict[str, Any],
) -> float:
    """
    Average the three layer scores into a single composite score (0–100).

    Layer score extraction:
      - data_forensics   → ``health_score``           (float, 0–100)
      - stress_test      → ``bias_score``             (float, 0–100)
      - fairness_metrics → ``overall_fairness_score`` (float, 0–100)

    Any missing / None score defaults to 50.0 (neutral) so a missing
    layer does not catastrophically skew the result; a warning is logged.
    """
    def _extract(result: Dict[str, Any], key: str, layer: str) -> float:
        val = result.get(key)
        if val is None:
            logger.warning(
                "Score key '%s' missing from '%s' result — defaulting to 50.0.",
                key, layer,
            )
            return 50.0
        try:
            return float(val)
        except (TypeError, ValueError):
            logger.warning(
                "Score key '%s' in '%s' is not numeric (%r) — defaulting to 50.0.",
                key, layer, val,
            )
            return 50.0

    df_score = _extract(data_forensics, "health_score", "data_forensics")
    st_score = _extract(stress_test, "bias_score", "stress_test")
    fm_score = _extract(fairness_metrics, "overall_fairness_score", "fairness_metrics")

    composite = (df_score + st_score + fm_score) / 3.0
    return round(composite, 2)


def _certification_status(score: float) -> str:
    """Map composite score to a certification tier."""
    if score >= _CERTIFIED_THRESHOLD:
        return "CERTIFIED"
    if score >= _CONDITIONAL_THRESHOLD:
        return "CONDITIONALLY CERTIFIED"
    return "NOT CERTIFIED"


# ---------------------------------------------------------------------------
# CertificateGenerator
# ---------------------------------------------------------------------------

class CertificateGenerator:
    """
    Issues, stores, and verifies ModelPassport bias-audit certificates.

    Thread-safety
    -------------
    A single ``threading.Lock`` guards all reads and writes to the
    counter file and the store file, making this class safe for use inside
    multi-threaded servers.

    Usage
    -----
    >>> gen = CertificateGenerator()
    >>> cert = gen.generate(
    ...     model_name="HiringClassifier-v3",
    ...     organization="Acme Corp",
    ...     domain="hiring",
    ...     data_forensics=df_result,
    ...     stress_test=st_result,
    ...     fairness_metrics=fm_result,
    ...     gemini_governance=gg_result,
    ... )
    >>> print(cert["certificate_id"])   # MP-2026-000001
    >>> retrieved = gen.verify("MP-2026-000001")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Counter management
    # ------------------------------------------------------------------

    def _next_certificate_id(self) -> str:
        """
        Atomically increment the counter and return the formatted ID.
        Must be called while ``self._lock`` is held.
        """
        counter_data: Dict[str, int] = _load_json_file(
            _COUNTER_FILE, {"counter": 0}
        )
        current = counter_data.get("counter", 0)
        next_val = current + 1
        _write_json_file(_COUNTER_FILE, {"counter": next_val})
        return f"MP-{_CERT_YEAR}-{next_val:06d}"

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def _load_store(self) -> Dict[str, Any]:
        """Load the certificate store. Must be called while lock is held."""
        return _load_json_file(_STORE_FILE, {})

    def _save_to_store(self, certificate: Dict[str, Any]) -> None:
        """Append a certificate to the store. Must be called while lock is held."""
        store = self._load_store()
        store[certificate["certificate_id"]] = certificate
        _write_json_file(_STORE_FILE, store)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        model_name: str,
        organization: str,
        domain: str,
        data_forensics: Dict[str, Any],
        stress_test: Dict[str, Any],
        fairness_metrics: Dict[str, Any],
        gemini_governance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Issue a new ModelPassport certificate from the four audit layer results.

        Parameters
        ----------
        model_name       : Identifying name of the audited model.
        organization     : Name of the submitting organisation.
        domain           : Application domain (e.g. "hiring", "lending").
        data_forensics   : Result dict from DataForensicsEngine.run().
        stress_test      : Result dict from StressTestEngine.run().
        fairness_metrics : Result dict from FairnessMetricsEngine.run().
        gemini_governance: Result dict from GeminiGovernanceEngine.run().

        Returns
        -------
        Certificate dict with the following exact keys:
          certificate_id       – e.g. "MP-2026-000001"
          sha256_hash          – SHA-256 of the full results payload
          issued_at            – UTC ISO-8601 timestamp
          model_name
          organization
          domain
          overall_score        – float in [0, 100]
          certification_status – "CERTIFIED" | "CONDITIONALLY CERTIFIED" | "NOT CERTIFIED"
          verify_url           – "https://modelpassport.ai/verify/{certificate_id}"
          layer_results        – dict containing all four raw layer result dicts
          narrative            – plain-language summary from Gemini governance layer
          remediation_checklist– List[str] of action items from Gemini governance layer
        """
        # ── Compute scores & status ──────────────────────────────────────────
        overall_score = _compute_overall_score(
            data_forensics, stress_test, fairness_metrics
        )
        cert_status = _certification_status(overall_score)

        # ── Assemble layer_results payload ───────────────────────────────────
        layer_results: Dict[str, Any] = {
            "data_forensics": data_forensics,
            "stress_test": stress_test,
            "fairness_metrics": fairness_metrics,
            "gemini_governance": gemini_governance,
        }

        # ── SHA-256 integrity hash ───────────────────────────────────────────
        # Hash covers the full results payload for tamper-evidence.
        sha256_hash = _compute_sha256(layer_results)

        # ── Extract Gemini narrative & remediation from governance layer ──────
        narrative: str = gemini_governance.get("narrative", "")
        remediation_checklist: List[str] = gemini_governance.get(
            "remediation_checklist", []
        )

        # ── Thread-safe ID allocation + persistence ──────────────────────────
        with self._lock:
            certificate_id = self._next_certificate_id()
            issued_at = datetime.now(timezone.utc).isoformat()

            certificate: Dict[str, Any] = {
                "certificate_id": certificate_id,
                "sha256_hash": sha256_hash,
                "issued_at": issued_at,
                "model_name": model_name,
                "organization": organization,
                "domain": domain,
                "overall_score": overall_score,
                "certification_status": cert_status,
                "verify_url": f"https://modelpassport.ai/verify/{certificate_id}",
                "layer_results": layer_results,
                "narrative": narrative,
                "remediation_checklist": remediation_checklist,
            }

            self._save_to_store(certificate)

        logger.info(
            "Certificate issued — id=%s, score=%.2f, status=%s, org=%s.",
            certificate_id, overall_score, cert_status, organization,
        )

        return certificate

    def verify(self, certificate_id: str) -> Dict[str, Any]:
        """
        Retrieve a previously issued certificate by its ID.

        Parameters
        ----------
        certificate_id : str
            The certificate ID to look up (e.g. "MP-2026-000001").

        Returns
        -------
        The full certificate dict as originally returned by ``generate()``.

        Raises
        ------
        ValueError
            If no certificate with the given ID exists in the store.
        """
        with self._lock:
            store = self._load_store()

        certificate = store.get(certificate_id)
        if certificate is None:
            raise ValueError(
                f"Certificate '{certificate_id}' not found. "
                "It may not exist or the store file may have been moved."
            )

        logger.info("Certificate verified — id=%s.", certificate_id)
        return certificate
