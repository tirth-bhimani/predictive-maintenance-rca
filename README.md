# Predictive Maintenance & Root Cause Analysis Platform

An end-to-end machine learning application that predicts the **Remaining Useful Life (RUL)** of industrial machines, classifies their failure risk, and explains each prediction using SHAP — served through a FastAPI backend and a Streamlit dashboard, fully containerized with Docker.

Built on the **NASA C-MAPSS FD001** turbofan engine dataset.

---

## Overview

Traditional maintenance reacts after a machine fails. This project takes the opposite approach: it continuously estimates how much useful life a machine has left, so maintenance can be planned before a failure happens.

```
Sensor Data → XGBoost Model → Predicted RUL → Risk Level → SHAP Explanation → Maintenance Decision
```

Given a machine's current sensor readings, the system returns:

- **Predicted RUL** — estimated operating cycles remaining
- **Risk level** — HIGH / MEDIUM / LOW, derived from the predicted RUL
- **Root cause factors** — the top 3 sensor features driving that specific prediction (via SHAP)

| Predicted RUL | Risk |
|---|---|
| < 20 cycles | HIGH |
| 20-49 cycles | MEDIUM |
| >= 50 cycles | LOW |

---

## Architecture

```
NASA C-MAPSS FD001
        |
        v
Processed Sensor Data
        |
        v
  Streamlit Dashboard  --POST /predict-->  FastAPI
                                              |
                                              v
                                       XGBoost RUL Model
                                              |
                                   +----------+----------+
                                   v          v          v
                                  RUL       Risk    SHAP Factors
```

The dashboard lets a user pick a machine and a cycle, sends that cycle's sensor data to the API, and displays the prediction, risk level, and top contributing factors — updating automatically as the cycle changes.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Modeling | XGBoost, SHAP, Pandas, NumPy |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Deployment | Docker, Docker Compose |

The current version is intentionally scoped to keep the core prediction pipeline simple — there's no database or auth layer yet (see Roadmap below).

---

## Project Structure

```
predictive-maintenance-rca/
├── data/processed/train_processed.csv
├── src/
│   ├── api/main.py                        # FastAPI app
│   ├── dashboard/streamlit_app.py          # Streamlit UI
│   └── models/xgboost_rul/xgboost_rul_model.json
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
└── requirements.txt
```

---

## Feature Set

The model is trained on 24 raw inputs (cycle, 3 operating settings, 21 sensors) plus engineered 10-cycle rolling mean/std for each sensor — 45+ features total, e.g. `sensor11`, `sensor11_mean_10`, `sensor11_std_10`.

---

## Running Locally

```bash
git clone https://github.com/tirth-bhimani/predictive-maintenance-rca.git
cd predictive-maintenance-rca
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Start the API** (terminal 1)
```bash
uvicorn src.api.main:app --reload
```
API: `http://127.0.0.1:8000` · Swagger docs: `http://127.0.0.1:8000/docs`

**Start the dashboard** (terminal 2)
```bash
streamlit run src/dashboard/streamlit_app.py
```
Dashboard: `http://localhost:8501`

---

## Running with Docker

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |

Two containers are built: `predictive-maintenance-api` and `predictive-maintenance-dashboard`, with the dashboard depending on the API. Inside Docker, the dashboard reaches the API via the service name (`http://api:8000`), not `localhost`.

---

## API

**`POST /predict`** — accepts a machine's current cycle + sensor readings, returns the prediction.

Request (abbreviated — full payload includes all 21 sensors and their rolling features):
```json
{
  "unit": 1,
  "cycle": 50,
  "setting1": 0.0, "setting2": 0.0, "setting3": 100.0,
  "sensor1": 518.67, "sensor11": 47.47,
  "sensor1_mean_10": 518.67, "sensor1_std_10": 0.5
}
```

Response:
```json
{
  "unit": 1,
  "predicted_rul": 42.75,
  "risk": "MEDIUM",
  "top_factors": [
    { "feature": "sensor11", "importance": 5.23 },
    { "feature": "sensor4", "importance": 4.87 },
    { "feature": "sensor15", "importance": 3.91 }
  ]
}
```

Other endpoints: `GET /` and `GET /health`.

---

## Design Notes

**Why RUL + risk level, not just risk level?**
A raw risk label ("HIGH") doesn't tell an engineer *why*. Pairing the numeric RUL with SHAP-ranked contributing sensors turns a black-box classification into something actionable — e.g. "RUL = 18, driven mainly by sensor11 and sensor4."

**Why no database yet?**
The current scope is deliberately a single prediction pipeline (data → model → API → dashboard) to keep the core ML/serving loop clean and easy to reason about. Persistence, auth, and monitoring are the next layer to add, not the first.

---

## Limitations

- Sensor input is simulated from the NASA dataset, not a live industrial feed
- Only the FD001 subset is used (single operating condition, single fault mode)
- Risk thresholds (20 / 50 cycles) are fixed defaults, not validated against real failure data
- No authentication, logging, or model versioning — not production-hardened

---

## License

Educational / portfolio project.