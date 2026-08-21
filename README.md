# Predictive Maintenance & Root Cause Analysis for Turbofan Engines

Machine learning system for predicting **Remaining Useful Life (RUL)** of turbofan engines and explaining *why* a given prediction was made, using NASA's C-MAPSS sensor dataset.

**Status:** In development — core ML pipeline complete, API/dashboard/deployment layer in progress.

---

## Summary

Unplanned equipment failure is expensive and disruptive. This project builds an ML pipeline that:

- Predicts how many operating cycles an engine has left before failure (RUL)
- Flags engines that are likely to fail soon, based on a configurable threshold
- Explains individual predictions using SHAP, identifying which sensor readings drove the result
- Compares a tree-based model (XGBoost) against a sequence model (LSTM) to justify the model choice
- Tracks all experiments (parameters, metrics) with MLflow for reproducibility

The goal was to build something closer to a real maintenance decision-support tool than a one-off notebook: proper train/validation splitting by engine unit, hyperparameter tuning, experiment tracking, and model interpretability, not just a single RMSE number.

---

## Dataset

**NASA C-MAPSS, subset FD001** — simulated turbofan engine degradation data.

- 100 training engine units, run to failure
- 3 operational settings + 21 sensor channels per cycle
- Target: Remaining Useful Life (cycles until failure)

---

## Approach

**1. Label construction**
RUL is computed per engine as `max_cycle − current_cycle`, then capped at 125 cycles. Capping prevents the model from over-focusing on engines that are still far from failure, and puts more training signal on the degradation window that actually matters operationally.

**2. Feature engineering**
Rolling mean and rolling standard deviation are computed over a 10-cycle window for each sensor, to capture trend and volatility rather than just instantaneous readings.

**3. Train/validation split**
Split by **engine unit ID**, not by row. This avoids leaking cycles from the same engine across both sets, which would otherwise give an unrealistically optimistic validation score.

**4. Modeling**
Two models were trained and compared:

| Model | Type | Role |
|---|---|---|
| XGBoost | Gradient-boosted trees on engineered tabular features | Primary model |
| LSTM (PyTorch) | Sequence model over 30-cycle sliding windows | Comparison baseline |

XGBoost hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`) were tuned with `RandomizedSearchCV`. On this feature set, XGBoost outperformed the LSTM on validation RMSE, so it was selected as the primary model — the LSTM is kept in the repo as a documented baseline for comparison, which was itself part of the point: showing that a more complex model isn't automatically the right choice.

**5. Failure flagging**
Predicted RUL below a configurable threshold (default: 20 cycles) is flagged as `failing_soon = True`.

**6. Explainability**
For any prediction, SHAP's `TreeExplainer` ranks feature contributions and returns the top 3 sensor/rolling features responsible for that prediction — turning "RUL = 18" into "RUL = 18, driven mainly by sensor_11 and rolling volatility in sensor_4," which is what an actual maintenance decision needs.

**7. Experiment tracking**
All runs (hyperparameters, RMSE, MAE, config) are logged with MLflow to a local `mlruns/` store for comparison across runs.

---

## Architecture

```
C-MAPSS FD001
      │
      ▼
Preprocessing (RUL calc + capping, rolling features)
      │
      ▼
Unit-based train/validation split
      │
   ┌──┴──┐
   ▼     ▼
XGBoost  LSTM   →  model comparison → XGBoost selected
   │
   ├── Failure flag (RUL < threshold)
   └── SHAP → Root Cause Analysis (top-3 features)
```

---

## Project Structure

```
predictive-maintenance-rca/
├── data/
│   ├── raw/            # not committed — see data/raw/README.md
│   └── processed/      # not committed — see data/processed/README.md
├── src/
│   ├── data/preprocessing.py
│   ├── models/
│   │   ├── xgboost_model.py
│   │   └── lstm.py
│   ├── rca/
│   │   ├── root_cause.py
│   │   └── test_rca.py
│   ├── api/             # in progress
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── db/models.py
│   └── dashboard/        # in progress
│       └── streamlit_app.py
├── docker/                # in progress
│   ├── Dockerfile.api
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

`venv/`, `mlruns/`, model checkpoints (`*.pth`, `*.pt`, `*.pkl`), and raw/processed data are gitignored to keep the repo lightweight.

---

## Tech Stack

**Core:** Python, Pandas, NumPy, Scikit-learn
**Modeling:** XGBoost, PyTorch
**Explainability:** SHAP
**Experiment tracking:** MLflow
**Planned (in progress):** FastAPI, Streamlit, SQLAlchemy, PostgreSQL, Docker

---

## Running It

```powershell
git clone https://github.com/tirth-bhimani/predictive-maintenance-rca.git
cd predictive-maintenance-rca

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Preprocess data**
```powershell
python src/data/preprocessing.py
```

**Train XGBoost** (tuning, evaluation, MLflow logging)
```powershell
python src/models/xgboost_model.py
```

**Train LSTM comparison model**
```powershell
python src/models/lstm.py
```

**Run Root Cause Analysis** (SHAP-based feature ranking on trained model)
```powershell
python src/rca/test_rca.py
```

**View experiment tracking**
```powershell
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --backend-store-uri ./mlruns
```
Then open `http://127.0.0.1:5000`.

---

## What's Done vs. What's Next

**Done**
- Data preprocessing, RUL labeling and capping, rolling feature engineering
- Unit-based train/validation split
- XGBoost model with hyperparameter tuning, RMSE/MAE evaluation
- LSTM baseline with early stopping, for model comparison
- Failure threshold flagging
- SHAP-based Root Cause Analysis (top-3 contributing features)
- MLflow experiment tracking

**Next**
- FastAPI prediction service
- PostgreSQL storage for predictions/history
- Streamlit dashboard for monitoring
- Dockerized deployment
- Data drift / model performance monitoring in production

---

## Why XGBoost Over LSTM

Both a tabular tree-based model and a sequence-based deep learning model were evaluated rather than assuming one was better. XGBoost performed better on this engineered feature set (rolling statistics already capture a lot of the temporal signal a sequence model would otherwise need to learn), so it was chosen as the primary model. The LSTM is kept in the repo as evidence of that comparison rather than as dead code.

---

## License

Educational / portfolio project.