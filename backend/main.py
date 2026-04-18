"""
backend/main.py

ModelPassport API — End-to-End Audit Pipeline
==============================================

Routes
------
POST /audit/full          — Full 4-layer audit pipeline (main entry point)
GET  /verify/{id}         — Retrieve & verify an issued certificate
GET  /health              — Liveness check

Kept as stubs (not removed, for backwards-compat):
POST /audit/data
POST /audit/stress-test
POST /audit/fairness
POST /audit/report
"""

import asyncio
import io
import logging
import uuid
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import uvicorn

from backend.config import settings
from backend.layers.data_forensics import DataForensicsEngine
from backend.layers.fairness_metrics import FairnessMetricsEngine
from backend.layers.gemini_governance import GeminiGovernanceEngine
from backend.layers.stress_test import StressTestEngine
from backend.utils.report_generator import CertificateGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ModelPassport API",
    description="Pre-deployment AI bias certification — 4-layer audit pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models (kept for backwards-compat with stubs)
# ---------------------------------------------------------------------------
class AuditRequest(BaseModel):
    model_name: str
    organization: str
    domain: str                   # e.g. "hiring", "loan", "healthcare"
    protected_attributes: list[str]


class ReportRequest(BaseModel):
    audit_id: str
    model_name: str
    organization: str
    layer_results: dict


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions and returns a clean JSON error.
    Never exposes raw tracebacks to the client.
    """
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": str(exc)},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Encode every object/categorical column in *df* with a LabelEncoder
    so the resulting frame contains only numeric values.

    Returns
    -------
    encoded_df  : numeric-only copy of *df*
    encoders    : map of column_name → fitted LabelEncoder (for reference)
    """
    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le
    return encoded, encoders


def _derive_bias_score(stress_test_result: dict) -> float:
    """
    Derive a 0–100 bias_score from StressTestEngine output so that
    CertificateGenerator._compute_overall_score() can consume it.

    Logic: average counterfactual flipping rate across all protected
    attributes, then convert to a 0–100 score where higher = less biased.
       bias_score = 100 × (1 − mean_flipping_rate)
    """
    attrs = stress_test_result.get("protected_attributes", {})
    if not attrs:
        return 50.0   # neutral fallback

    rates = [
        data.get("flipping_rate", 0.0)
        for data in attrs.values()
        if isinstance(data, dict)
    ]
    if not rates:
        return 50.0

    mean_flip = float(np.mean(rates))
    return round(max(0.0, min(100.0, (1.0 - mean_flip) * 100)), 2)


# ---------------------------------------------------------------------------
# Main route — POST /audit/full
# ---------------------------------------------------------------------------
@app.post("/audit/full")
async def audit_full(
    dataset_file: UploadFile = File(..., description="CSV dataset file"),
    model_file: Optional[UploadFile] = File(None, description="Joblib-serialised sklearn model (optional)"),
    model_name: str = Form(...),
    organization: str = Form(...),
    domain: str = Form(...),
    target_column: str = Form(...),
    protected_attributes: str = Form(..., description="Comma-separated list of protected attribute column names"),
):
    """
    Full end-to-end ModelPassport audit pipeline.

    1. Read CSV → pandas DataFrame
    2. Load or train an sklearn model
    3. 80/20 train/test split + encode categoricals for model ops
    4. Layer 1 — DataForensicsEngine   (raw DataFrame)
    5. Layer 2 — StressTestEngine      (model + encoded full DataFrame)
    6. Layer 3 — FairnessMetricsEngine (model + encoded test split with target)
    7. Layer 4 — GeminiGovernanceEngine (combined layer results)
    8. CertificateGenerator            → structured certificate dict
    9. Return full certificate
    """
    try:
        # ── 0. Parse inputs ─────────────────────────────────────────────────
        attrs: list[str] = [a.strip() for a in protected_attributes.split(",") if a.strip()]
        logger.info(
            "audit/full — model='%s', org='%s', domain='%s', target='%s', attrs=%s",
            model_name, organization, domain, target_column, attrs,
        )

        # ── 1. Read CSV ──────────────────────────────────────────────────────
        logger.info("Step 1: reading CSV dataset…")
        raw_bytes = await dataset_file.read()
        df_raw: pd.DataFrame = pd.read_csv(io.BytesIO(raw_bytes))
        logger.info("Dataset shape: %s", df_raw.shape)

        if target_column not in df_raw.columns:
            raise HTTPException(
                status_code=422,
                detail=f"target_column '{target_column}' not found in CSV. "
                       f"Available columns: {list(df_raw.columns)}",
            )

        for attr in attrs:
            if attr not in df_raw.columns:
                raise HTTPException(
                    status_code=422,
                    detail=f"Protected attribute '{attr}' not found in dataset columns: {list(df_raw.columns)}",
                )

        # ── 2-8: Run the entire pipeline under a timeout ────────────────────
        async def _run_pipeline():
            # ── 2. Load or train model ───────────────────────────────────────
            logger.info("Step 2: preparing model…")
            df_encoded, _ = _encode_dataframe(df_raw)

            y_full = df_encoded[target_column]
            X_full = df_encoded.drop(columns=[target_column])

            if model_file is not None:
                logger.info("Loading provided model from uploaded file…")
                model_bytes = await model_file.read()
                model = joblib.load(io.BytesIO(model_bytes))
                logger.info("Model loaded: %s", type(model).__name__)
            else:
                logger.info("No model provided — training fallback RandomForestClassifier…")
                X_train_fb, _, y_train_fb, _ = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42
                )
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_fb, y_train_fb)
                logger.info("Fallback RandomForest trained on %d samples.", len(X_train_fb))

            # ── 3. Train/test split (encoded) ────────────────────────────────
            logger.info("Step 3: creating 80/20 train/test split…")
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42
            )

            df_test_encoded = X_test.copy()
            df_test_encoded[target_column] = y_test.values

            # ── 4. Layer 1 — Data Forensics ──────────────────────────────────
            logger.info("Step 4: running DataForensicsEngine…")
            df_layer = DataForensicsEngine(
                df=df_raw,
                protected_attributes=attrs,
                target_column=target_column,
            )
            forensics_result = df_layer.run()
            logger.info(
                "Data Forensics complete — health_score=%s, status=%s",
                forensics_result.get("health_score"), forensics_result.get("status"),
            )

            # ── 5. Layer 2 — Stress Test ─────────────────────────────────────
            logger.info("Step 5: running StressTestEngine…")
            stress_engine = StressTestEngine(
                model=model,
                dataset=df_encoded,
                protected_attributes=attrs,
                target_column=target_column,
            )
            stress_result = stress_engine.run(num_samples=500)
            stress_result["bias_score"] = _derive_bias_score(stress_result)
            logger.info(
                "Stress Test complete — bias_score=%s, status=%s",
                stress_result.get("bias_score"), stress_result.get("status"),
            )

            # ── 6. Layer 3 — Fairness Metrics ────────────────────────────────
            logger.info("Step 6: running FairnessMetricsEngine…")
            fairness_engine = FairnessMetricsEngine(
                model=model,
                dataset=df_test_encoded,
                protected_attributes=attrs,
                target_column=target_column,
            )
            fairness_result = fairness_engine.run()
            logger.info(
                "Fairness Metrics complete — overall_fairness_score=%s, status=%s",
                fairness_result.get("overall_fairness_score"), fairness_result.get("status"),
            )

            # ── 7. Layer 4 — Gemini Governance ───────────────────────────────
            logger.info("Step 7: running GeminiGovernanceEngine…")
            combined_for_gemini = {
                "data_forensics": forensics_result,
                "stress_test": stress_result,
                "fairness_metrics": fairness_result,
            }
            gemini_engine = GeminiGovernanceEngine()
            governance_result = gemini_engine.run(combined_for_gemini)
            logger.info(
                "Gemini Governance complete — risk=%s, gemini_status=%s",
                governance_result.get("severity_summary", {}).get("overall_risk_level"),
                governance_result.get("gemini_status"),
            )

            # ── 8. Generate Certificate ───────────────────────────────────────
            logger.info("Step 8: generating certificate…")
            generator = CertificateGenerator()
            certificate = generator.generate(
                model_name=model_name,
                organization=organization,
                domain=domain,
                data_forensics=forensics_result,
                stress_test=stress_result,
                fairness_metrics=fairness_result,
                gemini_governance=governance_result,
            )
            logger.info(
                "Certificate issued — id=%s, score=%s, status=%s",
                certificate.get("certificate_id"),
                certificate.get("overall_score"),
                certificate.get("certification_status"),
            )
            return certificate

        try:
            certificate = await asyncio.wait_for(_run_pipeline(), timeout=300)
        except asyncio.TimeoutError:
            logger.error("audit/full pipeline timed out after 300 seconds.")
            raise HTTPException(
                status_code=504,
                detail="Audit timed out after 300 seconds. Try a smaller dataset.",
            )

        return certificate

    except HTTPException:
        # Re-raise FastAPI validation errors unchanged
        raise
    except Exception as exc:
        logger.error("audit/full pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Audit pipeline failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /verify/{certificate_id}
# ---------------------------------------------------------------------------
@app.get("/verify/{certificate_id}")
async def verify_certificate(certificate_id: str):
    """
    Retrieve and verify an issued ModelPassport certificate by its ID.
    Returns the full certificate dict or 404 if the ID is unknown.
    """
    logger.info("Verification requested for certificate: %s", certificate_id)
    try:
        certificate = CertificateGenerator().verify(certificate_id)
        return certificate
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Certificate '{certificate_id}' not found.",
        )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Health check endpoint to verify API readiness.
    """
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Legacy stub routes (kept for backwards-compatibility)
# ---------------------------------------------------------------------------
@app.post("/audit/data", tags=["deprecated"])
async def audit_data(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    organization: str = Form(...),
    domain: str = Form(...),
    protected_attributes: list[str] = Form(...),
):
    """Deprecated — use POST /audit/full instead."""
    return {
        "audit_id": str(uuid.uuid4()),
        "status": "deprecated",
        "message": "This route is a stub. Use POST /audit/full for the full pipeline.",
    }


@app.post("/audit/stress-test", tags=["deprecated"])
async def audit_stress_test(request: AuditRequest):
    """Deprecated — use POST /audit/full instead."""
    return {
        "audit_id": str(uuid.uuid4()),
        "status": "deprecated",
        "message": "This route is a stub. Use POST /audit/full for the full pipeline.",
    }


@app.post("/audit/fairness", tags=["deprecated"])
async def audit_fairness(request: AuditRequest):
    """Deprecated — use POST /audit/full instead."""
    return {
        "audit_id": str(uuid.uuid4()),
        "status": "deprecated",
        "message": "This route is a stub. Use POST /audit/full for the full pipeline.",
    }


@app.post("/audit/report", tags=["deprecated"])
async def audit_report(request: ReportRequest):
    """Deprecated — use POST /audit/full instead."""
    return {
        "certificate_id": str(uuid.uuid4()),
        "status": "deprecated",
        "message": "This route is a stub. Use POST /audit/full for the full pipeline.",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)
