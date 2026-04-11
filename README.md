# ModelPassport

> **The Safety Certificate for Government AI**

ModelPassport is a **pre-deployment AI bias certification platform** built for the [Google Solution Challenge 2026](https://developers.google.com/community/gdsc-solution-challenge). It audits machine learning models across four automated layers before they go live in government and institutional settings — issuing tamper-evident, publicly verifiable certificates that prove a model has passed independent fairness scrutiny.

---

## Why it matters

AI systems deployed in hiring, lending, healthcare, and public services can reproduce or amplify historical biases at scale. ModelPassport gives institutions a structured, automated way to certify that a model meets fairness thresholds *before* it is allowed to make decisions affecting real people.

---

## How it works

A model and its test dataset are submitted to ModelPassport. Four audit engines run in sequence, and their combined results are passed to a certificate generator that issues a unique, SHA-256-signed certification document.

```
Dataset + Model
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Layer 1 · Data Forensics                       │
│  Layer 2 · Synthetic Stress Test                │
│  Layer 3 · Fairness Metrics                     │
│  Layer 4 · Gemini Governance (Gemini 1.5 Flash) │
└─────────────────────────────────────────────────┘
      │
      ▼
 Certificate  MP-2026-000001
 SHA-256 signed · Publicly verifiable
```

---

## The Four Audit Layers

### 1 · Data Forensics
Analyses the raw dataset for structural health issues — class imbalance, missing values, representation gaps across protected groups, and statistical distribution anomalies. Returns a `health_score` (0–100) and a list of human-readable warnings.

### 2 · Synthetic Stress Test
Generates counterfactual *twin profiles* — identical individuals who differ only in a protected attribute (e.g. gender, race, age group). Runs both twins through the model and measures how often the prediction flips. Also enforces the EEOC **80% Four-Fifths Disparate Impact Rule** across all protected groups. Returns a `bias_score` and per-attribute flipping rates.

### 3 · Fairness Metrics
Computes four established fairness dimensions using `fairlearn` and `scikit-learn`:
- **Demographic Parity Difference** — gap in positive prediction rates across groups
- **Equalized Odds Difference** — gap in true positive rates across groups
- **Disparate Impact Ratio** — min/max positive rate ratio (80% rule)
- **Individual Fairness Score** — Lipschitz-inspired nearest-neighbour consistency check

Returns an `overall_fairness_score` (0–100) and per-attribute metric breakdowns.

### 4 · Gemini Governance
Feeds the combined results from all three upstream layers to **Google Gemini 1.5 Flash**. Gemini produces:
- A plain-language **narrative** explaining the findings for non-technical policymakers
- A prioritised **remediation checklist** of concrete actions to address detected issues
- A **severity summary** classifying the model's overall risk level as `high`, `medium`, or `low`

---

## Certificate System

Every audit that completes successfully produces a ModelPassport certificate:

| Field | Example |
|---|---|
| `certificate_id` | `MP-2026-000001` |
| `sha256_hash` | SHA-256 of the full results payload |
| `issued_at` | `2026-04-11T15:30:00+00:00` |
| `certification_status` | `CERTIFIED` / `CONDITIONALLY CERTIFIED` / `NOT CERTIFIED` |
| `overall_score` | `84.73` (average of three layer scores) |
| `verify_url` | `https://modelpassport.ai/verify/MP-2026-000001` |

**Certification thresholds:**
- `overall_score >= 80` → **CERTIFIED**
- `overall_score >= 60` → **CONDITIONALLY CERTIFIED**
- `overall_score < 60`  → **NOT CERTIFIED**

Certificates are stored locally in `backend/certificates_store.json` with sequential, auto-incrementing IDs (`MP-2026-000001`, `MP-2026-000002`, …) and can be retrieved at any time via the `/verify/{certificate_id}` API endpoint.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| API Framework | FastAPI + Uvicorn |
| ML / Fairness | scikit-learn, fairlearn |
| Governance AI | Google Gemini API (`gemini-1.5-flash`) |
| Deployment | Google Cloud Run |
| Frontend | Vanilla HTML / CSS / JavaScript |

---

## Project Structure

```
ModelPassport/
├── backend/
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── data_forensics.py        # Layer 1 — DataForensicsEngine
│   │   ├── stress_test.py           # Layer 2 — StressTestEngine
│   │   ├── fairness_metrics.py      # Layer 3 — FairnessMetricsEngine
│   │   └── gemini_governance.py     # Layer 4 — GeminiGovernanceEngine
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py         # Sample dataset utilities
│   │   └── report_generator.py      # CertificateGenerator
│   ├── config.py                    # App settings (env vars)
│   ├── main.py                      # FastAPI app + routes
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── sample_data/
│   └── README.md                    # Sample dataset instructions
├── .env.example                     # Environment variable template
├── .gitignore
├── Dockerfile
├── PROGRESS.md                      # Development log & task tracker
└── README.md
```

---

## Setup & Running Locally

> **Prerequisites:** Python 3.14, a Google Gemini API key

### 1 · Clone the repository

```bash
git clone https://github.com/your-org/ModelPassport.git
cd ModelPassport
```

### 2 · Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3 · Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4 · Run the API server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080`.  
Interactive docs: `http://localhost:8080/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/audit/data` | Run the Data Forensics layer (multipart/form-data with CSV) |
| `POST` | `/audit/stress-test` | Run the Synthetic Stress Test layer |
| `POST` | `/audit/fairness` | Run the Fairness Metrics layer |
| `POST` | `/audit/report` | Generate the Gemini governance report & certificate |
| `GET` | `/verify/{certificate_id}` | Retrieve and verify an issued certificate |
| `GET` | `/health` | Health check |

---

## Built For

**Google Solution Challenge 2026** — built by developers committed to responsible AI deployment in public institutions.

---

*ModelPassport — because every AI that governs people deserves independent scrutiny.*
