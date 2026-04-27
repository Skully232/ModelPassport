"""
backend/layers/gemini_governance.py

GeminiGovernanceEngine — ModelPassport Governance Reporting Layer

Accepts aggregated results from the three upstream audit layers:
  - data_forensics   (DataForensicsEngine)
  - stress_test      (StressTestEngine)
  - fairness_metrics (FairnessMetricsEngine)

Uses Google Gemini (gemini-1.5-flash) to synthesise a plain-language
governance report comprising:
  - narrative        : prose explanation suitable for non-technical policymakers
  - remediation_checklist : ordered action items to address identified issues
  - severity_summary : counts and overall risk level derived from warnings
  - gemini_status    : "ok" | "error" — reflects Gemini API call health
  - status           : "Pass" | "Fail" — based on presence of critical warnings
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Tuple

from google import genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

_CRITICAL_PREFIX = "[CRITICAL]"
_WARNING_PREFIX = "[WARNING]"


def _classify_warnings(
    combined_results: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Walk all three layer result dicts and collect every warning string,
    partitioning them into critical and standard buckets.

    Returns
    -------
    critical_warnings : List[str]
    standard_warnings : List[str]
    """
    critical: List[str] = []
    standard: List[str] = []

    # ── data_forensics ──────────────────────────────────────────────────────
    df_result = combined_results.get("data_forensics", {})
    for w in df_result.get("warnings", []):
        (critical if _CRITICAL_PREFIX in w.upper() else standard).append(w)

    # ── stress_test ─────────────────────────────────────────────────────────
    st_result = combined_results.get("stress_test", {})
    for w in st_result.get("overall_warnings", []):
        (critical if _CRITICAL_PREFIX in w.upper() else standard).append(w)
    # Also dig into per-attribute warnings from stress test
    for attr_data in st_result.get("protected_attributes", {}).values():
        if isinstance(attr_data, dict):
            for w in attr_data.get("warnings", []):
                (critical if _CRITICAL_PREFIX in w.upper() else standard).append(w)

    # ── fairness_metrics ────────────────────────────────────────────────────
    fm_result = combined_results.get("fairness_metrics", {})
    for w in fm_result.get("warnings", []):
        (critical if _CRITICAL_PREFIX in w.upper() else standard).append(w)

    # Deduplicate while preserving order
    seen: set = set()
    def dedup(lst: List[str]) -> List[str]:
        out = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return dedup(critical), dedup(standard)


def _build_severity_summary(
    critical_warnings: List[str],
    standard_warnings: List[str],
) -> Dict[str, Any]:
    """
    Compute a structured severity summary from the warning sets.

    Risk level logic:
      - high   : any critical warnings present
      - medium : no critical warnings but ≥ 1 standard warning
      - low    : no warnings at all
    """
    total_critical = len(critical_warnings)
    total_warnings = len(standard_warnings)

    if total_critical > 0:
        risk_level = "high"
    elif total_warnings > 0:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "total_critical_warnings": total_critical,
        "total_warnings": total_warnings,
        "overall_risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(combined_results: Dict[str, Any]) -> str:
    """
    Synthesise a rich, structured prompt from the three audit layer results,
    surfacing scores, statuses, and all warnings in a format that allows Gemini
    to produce a clear, actionable governance report.
    """
    df = combined_results.get("data_forensics", {})
    st = combined_results.get("stress_test", {})
    fm = combined_results.get("fairness_metrics", {})

    # ── Data Forensics section ───────────────────────────────────────────────
    df_status = df.get("status", "Unknown")
    df_warnings: List[str] = df.get("warnings", [])
    df_stats = df.get("summary_statistics", df.get("dataset_summary", {}))

    df_block = textwrap.dedent(f"""
        ## Data Forensics Audit
        Status : {df_status}
        Warnings ({len(df_warnings)}):
        {chr(10).join(f'  - {w}' for w in df_warnings) if df_warnings else '  None'}
        Summary statistics snapshot:
        {json.dumps(df_stats, indent=2, default=str) if df_stats else '  Not available'}
    """).strip()

    # ── Stress Test section ──────────────────────────────────────────────────
    st_status = st.get("status", "Unknown")
    st_overall_warnings: List[str] = st.get("overall_warnings", [])
    st_attrs = st.get("protected_attributes", {})

    attr_lines: List[str] = []
    for attr, data in st_attrs.items():
        if not isinstance(data, dict):
            continue
        flip_rate = data.get("flipping_rate", "N/A")
        di_ratio = data.get("disparate_impact_ratio_min_max", "N/A")
        attr_lines.append(
            f"  [{attr}] flipping_rate={flip_rate}, disparate_impact_ratio={di_ratio}"
        )

    st_block = textwrap.dedent(f"""
        ## Stress Test Audit
        Status : {st_status}
        Overall Warnings ({len(st_overall_warnings)}):
        {chr(10).join(f'  - {w}' for w in st_overall_warnings) if st_overall_warnings else '  None'}
        Per-Attribute Metrics:
        {chr(10).join(attr_lines) if attr_lines else '  None'}
    """).strip()

    # ── Fairness Metrics section ─────────────────────────────────────────────
    fm_status = fm.get("status", "Unknown")
    fm_score = fm.get("overall_fairness_score", "N/A")
    fm_warnings: List[str] = fm.get("warnings", [])
    fm_summary = fm.get("metrics_summary", {})

    fm_block = textwrap.dedent(f"""
        ## Fairness Metrics Audit
        Status                : {fm_status}
        Overall Fairness Score: {fm_score} / 100
        Warnings ({len(fm_warnings)}):
        {chr(10).join(f'  - {w}' for w in fm_warnings) if fm_warnings else '  None'}
        Metrics Summary (per protected attribute):
        {json.dumps(fm_summary, indent=2, default=str) if fm_summary else '  Not available'}
    """).strip()

    # ── Full prompt ──────────────────────────────────────────────────────────
    prompt = textwrap.dedent(f"""
        You are an AI governance expert preparing a bias certification report for a pre-deployment model audit.
        The following results come from three automated audit subsystems.
        Your audience is non-technical policymakers and compliance officers — avoid jargon.

        {df_block}

        {st_block}

        {fm_block}

        ---

        Based on the above audit results, generate a report in the EXACT format below.
        Do NOT add any text outside these two sections.

        === NARRATIVE ===
        Write 3–5 paragraphs that:
          1. Summarise the overall fairness health of the model.
          2. Explain any critical issues in plain language, referencing specific attributes.
          3. Describe the implications for real-world deployment.
          4. Provide context for any warnings, even if minor.
          5. End with an overall recommendation: approve, approve with conditions, or reject.

        === REMEDIATION CHECKLIST ===
        List concrete, prioritised action items as a numbered list.
        Each item must be actionable and specific (e.g., "Re-sample training data to balance <attribute>").
        If no issues were detected, write a single item: "No remediation required — model meets all fairness thresholds."
    """).strip()

    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_gemini_response(raw_text: str) -> Tuple[str, List[str]]:
    """
    Extract narrative and remediation checklist from Gemini's response text.

    Expected format (enforced by prompt):
        === NARRATIVE ===
        <prose>

        === REMEDIATION CHECKLIST ===
        1. item one
        2. item two
        ...

    Falls back gracefully if the format is not perfectly followed.
    """
    narrative = ""
    checklist: List[str] = []

    # Split on section headers
    narrative_match = re.search(
        r"===\s*NARRATIVE\s*===(.*?)(?===\s*REMEDIATION CHECKLIST\s*===|$)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    checklist_match = re.search(
        r"===\s*REMEDIATION CHECKLIST\s*===(.*?)$",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )

    if narrative_match:
        narrative = narrative_match.group(1).strip()
    else:
        # Fallback: use the entire response as narrative
        narrative = raw_text.strip()

    if checklist_match:
        checklist_raw = checklist_match.group(1).strip()
        # Extract numbered list items
        items = re.findall(r"^\s*\d+[\.\)]\s+(.+)", checklist_raw, re.MULTILINE)
        if items:
            checklist = [item.strip() for item in items if item.strip()]
        else:
            # Fallback: treat non-empty lines as checklist items
            checklist = [
                line.lstrip("-•* ").strip()
                for line in checklist_raw.splitlines()
                if line.strip()
            ]

    return narrative, checklist


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class GeminiGovernanceEngine:
    """
    GeminiGovernanceEngine — Governance Reporting via Google Gemini

    Accepts combined audit results from the three ModelPassport audit layers
    and uses Gemini (gemini-1.5-flash) to produce a structured plain-language
    governance report for policymakers.

    Parameters
    ----------
    api_key : str, optional
        Gemini API key. If not provided, falls back to the ``GEMINI_API_KEY``
        environment variable.
    model_name : str, optional
        Gemini model identifier (default: ``"gemini-1.5-flash"``).
    temperature : float, optional
        Sampling temperature for Gemini generation (default: ``0.4``).
        Lower values produce more deterministic, policy-appropriate prose.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.4,
    ) -> None:
        from dotenv import load_dotenv
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set the GEMINI_API_KEY environment "
                "variable or pass api_key= to GeminiGovernanceEngine()."
            )

        self._client = genai.Client(api_key=self.api_key)
        logger.info(
            "GeminiGovernanceEngine initialised with model '%s'.", self.model_name
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, combined_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the governance report from combined audit layer results.

        Parameters
        ----------
        combined_results : dict
            Must contain at least the following keys (each maps to a layer
            result dict):
              - ``data_forensics``
              - ``stress_test``
              - ``fairness_metrics``

        Returns
        -------
        dict with keys:
            narrative             : str  — plain-language prose for policymakers
            remediation_checklist : List[str] — ordered action items
            severity_summary      : dict — warning counts and overall risk level
            gemini_status         : "ok" | "error"
            status                : "Pass" | "Fail"
        """
        logger.info("GeminiGovernanceEngine: classifying warnings from all layers.")
        critical_warnings, standard_warnings = _classify_warnings(combined_results)
        severity_summary = _build_severity_summary(critical_warnings, standard_warnings)

        logger.info(
            "Severity summary — critical: %d, warnings: %d, risk: %s",
            severity_summary["total_critical_warnings"],
            severity_summary["total_warnings"],
            severity_summary["overall_risk_level"],
        )

        status = "Fail" if critical_warnings else "Pass"

        # ── Call Gemini ──────────────────────────────────────────────────────
        prompt = _build_prompt(combined_results)
        narrative = ""
        remediation_checklist: List[str] = []
        gemini_status = "ok"

        try:
            logger.info(
                "Sending governance prompt to Gemini (%s).", self.model_name
            )
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            raw_text = response.text
            logger.debug("Gemini raw response length: %d chars.", len(raw_text))

            narrative, remediation_checklist = _parse_gemini_response(raw_text)

            if not narrative:
                logger.warning(
                    "Gemini returned an empty narrative. Using fallback message."
                )
                narrative = (
                    "The Gemini governance model returned a response but no narrative "
                    "could be extracted. Please review the raw audit results directly."
                )

            if not remediation_checklist:
                remediation_checklist = [
                    "Review the raw audit results and consult a fairness expert."
                ]

        except Exception as exc:
            logger.error(
                "GeminiGovernanceEngine: Gemini API call failed — %s", exc
            )
            gemini_status = "error"
            narrative = (
                f"Governance narrative could not be generated due to an API error: "
                f"{exc}. Please review the raw audit layer results directly and "
                f"consult a fairness expert for manual assessment."
            )
            remediation_checklist = [
                "Resolve the Gemini API connectivity issue and re-run the governance report.",
                "In the interim, manually review all warnings surfaced by the audit layers.",
            ]

        return {
            "narrative": narrative,
            "remediation_checklist": remediation_checklist,
            "severity_summary": severity_summary,
            "gemini_status": gemini_status,
            "status": status,
        }
