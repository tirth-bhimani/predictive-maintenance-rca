# Predictive Maintenance & Root Cause Analysis

AI/ML system for **Remaining Useful Life (RUL) prediction and Root Cause Analysis** using engine sensor data.

> **Status:** 🚧 In Development

## Current Progress

### Data & Preprocessing
- NASA C-MAPSS FD001 dataset integrated
- RUL calculated from engine cycle history
- RUL capped at 125 cycles
- Rolling Mean and Standard Deviation features added
- Rolling window: 10 cycles
- Train/validation split performed by engine ID
- Processed datasets generated for model development

### XGBoost Model
- XGBoost Regressor implemented for RUL prediction
- Hyperparameter tuning using RandomizedSearchCV
- Model evaluated using RMSE and MAE
- Failure threshold added: predicted RUL < 20 cycles
- Trained model saved using XGBoost native JSON format

## Current Pipeline

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
XGBoost RUL Prediction
        ↓
Failure Prediction

## Dataset

NASA C-MAPSS FD001 turbofan engine dataset.

- 100 training engines
- 3 operating settings
- 21 sensor measurements
- 26 original columns
- RUL prediction target

## Project Structure

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
│   │   └── xgboost_model.py
│   └── rca/
├── requirements.txt
└── README.md

## Current Results

- Validation RMSE: Add after final evaluation
- Validation MAE: Add after final evaluation
- Failure threshold: RUL < 20 cycles

## Next Steps

- [ ] LSTM RUL prediction
- [ ] Model comparison
- [ ] SHAP explainability
- [ ] Root Cause Analysis
- [ ] Maintenance recommendations
- [ ] FastAPI
- [ ] Streamlit dashboard
- [ ] MLflow
- [ ] Docker

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · PyTorch · SHAP · FastAPI · Streamlit · PostgreSQL · MLflow · Docker