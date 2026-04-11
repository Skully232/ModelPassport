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

---

## In Progress

- [ ] Wire all layers into `main.py` routes
- [ ] Sample dataset integration
- [ ] Frontend UI
- [ ] Google Cloud Run deployment
- [ ] Demo video
- [ ] Submission deck

---

## Known Issues / Decisions Pending

- Certificate storage is local JSON — needs Cloud Firestore for production
- Frontend is static HTML/JS for prototype — needs React for production
- No authentication yet — API key auth planned post-hackathon
