# ModelPassport — Development Progress Log

## Project Info

| Field | Detail |
|---|---|
| Start date | April 2026 |
| Developer | Solo |
| Challenge | Google Solution Challenge 2026 — Unbiased AI Decision theme |
| Submission deadline | 15 days from start |

---

## Tech Decisions Log

| Component | Choice | Reason |
|---|---|---|
| Backend | FastAPI | Async support and automatic API docs |
| Python version | 3.14 | — |
| AI layer | Google Gemini 1.5 Flash | Free tier, plain language report generation |
| Fairness library | fairlearn | Industry standard, Microsoft-backed |
| Deployment target | Google Cloud Run | Serverless, free tier eligible |
| Certificate storage | Local JSON files for prototype | Designed to migrate to Cloud Firestore |
| Certificate ID format | MP-2026-XXXXXX | Sequential with `threading.Lock` for safety |

---

## Completed

- [x] Project scaffold — FastAPI, folder structure, Dockerfile
- [x] `config.py` — pydantic-settings, env validation
- [x] `data_forensics.py` — representation audit, proxy detection, class imbalance, health score
- [x] `stress_test.py` — synthetic twin generation, counterfactual flipping rate, disparate impact
- [x] `fairness_metrics.py` — demographic parity, equalized odds, disparate impact, individual fairness
- [x] `gemini_governance.py` — Gemini API integration, narrative generation, remediation checklist
- [x] `report_generator.py` — sequential certificates, SHA-256 hashing, store and verify
- [x] Full pipeline wired in main.py — all 4 layers connected end to end
- [x] Server runs locally on port 8080
- [x] Fixed Gemini library deprecation — migrated from google-generativeai to google-genai
- [x] Health endpoint tested and confirmed working locally
- [x] Sample dataset added — UCI Adult Income (adult.csv) in sample_data/
- [x] Full pipeline end-to-end test passed with UCI Adult Income dataset
- [x] Certificate MP-2026-000001 issued successfully
- [x] overall_score: 57.23, certification_status: NOT CERTIFIED — bias correctly detected
- [x] Fixed GeminiGovernanceEngine env loading issue
- [x] Certificate storage verified — certificate_counter.json and certificates_store.json working
- [x] GET /verify/{certificate_id} endpoint tested and confirmed working
- [x] Full backend pipeline production-ready and tested end to end
- [x] All 7 frontend pages built — verify, how-it-works, about, pricing, privacy, terms, api-docs
- [x] Backend input validation added — protected attributes checked against CSV columns (422 with exact column list)
- [x] 300 second timeout added to audit pipeline (asyncio.wait_for, returns 504 on timeout)
- [x] Package `__init__.py` files created — backend, backend/layers, backend/utils

---

## In Progress

- [ ] Google Cloud Run deployment in progress
- [ ] Need to set GEMINI_API_KEY as Cloud Run secret
- [ ] Demo video
- [ ] Submission deck

---

## Known Issues / Decisions Pending

- Certificate storage is local JSON — needs Cloud Firestore for production
- Frontend is static HTML/JS for prototype — needs React for production
- No authentication yet — API key auth planned post-hackathon
- Windows Smart App Control may block Python DLLs — run from terminal not Store Python
- Use `py` command instead of `python` on Windows
