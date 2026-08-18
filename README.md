# Predictive Maintenance & Root Cause Analysis

AI/ML system for **Remaining Useful Life (RUL) prediction and Root Cause Analysis** using time-series engine sensor data.

> **Status:** 🚧 In Development

## Current Progress

### Data & Preprocessing

* NASA C-MAPSS **FD001** dataset integrated
* RUL calculated from engine cycle history
* RUL capped at **125 cycles**
* Rolling Mean and Standard Deviation features added
* Rolling window: **10 cycles**
* Train/validation split performed by **engine ID** to prevent data leakage
* Processed datasets generated for model development

## Current Pipeline

```text
NASA C-MAPSS FD001
        ↓
Data Preprocessing
        ↓
RUL Calculation
        ↓
Rolling Features
        ↓
Unit-Based Train/Validation Split
        ↓
Processed Data
        ↓
ML / DL Models
```

## Dataset

NASA C-MAPSS FD001 turbofan engine dataset.

* 100 training engines
* 3 operating settings
* 21 sensor measurements
* 26 original columns
* RUL prediction target

## Project Structure

```text
ML_MINI/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   │   └── preprocessing.py
│   ├── models/
│   └── rca/
├── requirements.txt
└── README.md
```

## Next Steps

* [ ] XGBoost RUL prediction
* [ ] LSTM RUL prediction
* [ ] Model evaluation
* [ ] SHAP explainability
* [ ] Root Cause Analysis
* [ ] Maintenance recommendations
* [ ] FastAPI
* [ ] Streamlit dashboard
* [ ] MLflow
* [ ] Docker

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · PyTorch · SHAP · FastAPI · Streamlit · PostgreSQL · MLflow · Docker**
