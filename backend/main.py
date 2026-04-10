import uuid
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from backend.config import settings

# 3. Basic Python logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ModelPassport API")

# 2. CORS middleware allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Pydantic input models
class AuditRequest(BaseModel):
    model_name: str
    organization: str
    domain: str  # e.g. "hiring", "loan", "healthcare"
    protected_attributes: list[str]  # e.g. ["gender", "age", "religion"]

class ReportRequest(BaseModel):
    audit_id: str
    model_name: str
    organization: str
    layer_results: dict

# 6. Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions and returns a clean JSON error.
    Never exposes raw tracebacks to the client.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )

# 5. Routes
@app.post("/audit/data")
async def audit_data(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    organization: str = Form(...),
    domain: str = Form(...),
    protected_attributes: list[str] = Form(...)
):
    """
    Accepts a dataset (CSV) and AuditRequest metadata as form data.
    Returns a generated audit_id and initiates the data_forensics layer.
    """
    audit_id = str(uuid.uuid4())
    logger.info(f"Received data audit request for model: {model_name} (Audit ID: {audit_id})")
    
    return {
        "audit_id": audit_id, 
        "status": "received", 
        "layer": "data_forensics"
    }

@app.post("/audit/stress-test")
async def audit_stress_test(request: AuditRequest):
    """
    Receives an AuditRequest JSON body and initiates the stress testing evaluation.
    """
    audit_id = str(uuid.uuid4())
    logger.info(f"Received stress test request for model: {request.model_name}")
    
    return {
        "audit_id": audit_id,
        "status": "stub",
        "layer": "stress_test"
    }

@app.post("/audit/fairness")
async def audit_fairness(request: AuditRequest):
    """
    Receives an AuditRequest JSON body and evaluates fairness metrics.
    """
    audit_id = str(uuid.uuid4())
    logger.info(f"Received fairness audit request for model: {request.model_name}")
    
    return {
        "audit_id": audit_id,
        "status": "stub",
        "layer": "fairness_metrics"
    }

@app.post("/audit/report")
async def audit_report(request: ReportRequest):
    """
    Receives a ReportRequest JSON body containing layer results, and generates a compliance report.
    Returns a generated certificate_id.
    """
    certificate_id = str(uuid.uuid4())
    logger.info(f"Received report generation request for audit ID: {request.audit_id}")
    
    return {
        "certificate_id": certificate_id,
        "status": "stub",
        "layer": "report_generator"
    }

@app.get("/health")
async def health():
    """
    Health check endpoint to verify API prototype readiness.
    """
    return {
        "status": "ok", 
        "version": "1.0.0"
    }

@app.get("/verify/{certificate_id}")
async def verify_certificate(certificate_id: str):
    """
    Verifies the compliance details of an existing certificate_id.
    """
    logger.info(f"Look up requested for Certificate ID: {certificate_id}")
    return {
        "certificate_id": certificate_id,
        "status": "valid",
        "lookup": "stub"
    }

# 7. Add this at bottom
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8080, reload=True)
