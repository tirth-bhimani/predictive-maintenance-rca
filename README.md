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
- Selected as the current primary model

### LSTM Model

- Lightweight LSTM implemented using PyTorch
- 30-cycle sliding windows used for sequence generation
- Sequences created separately for each engine
- 1-2 layer LSTM architecture used
- Early stopping implemented
- LSTM checkpoint saved for later use
- LSTM evaluated using validation RMSE

### Model Comparison

- XGBoost and LSTM evaluated on the same validation split
- XGBoost currently performs better than the lightweight LSTM
- XGBoost selected as the primary RUL prediction model
- LSTM retained as a comparison model

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
   ┌───────────────┐
   │               │
   ↓               ↓
XGBoost          LSTM
   │               │
   ↓               ↓
RUL Prediction   RUL Prediction
   │               │
   └───────┬───────┘
           ↓
     Model Comparison
           ↓
   XGBoost Selected
   as Primary Model
           ↓
    Failure Prediction

## Dataset

NASA C-MAPSS FD001 turbofan engine dataset.

- 100 training engines
- 100 test engines
- 3 operating settings
- 21 sensor measurements
- 26 original columns
- RUL prediction target

## RUL Calculation

RUL is calculated using:

RUL = Maximum Cycle for Engine - Current Cycle

The RUL value is capped at 125 cycles to reduce the influence of very high early-life RUL values.

## Feature Engineering

The following time-series features are currently used:

- Rolling Mean
- Rolling Standard Deviation
- Rolling window size: 10 cycles

These features help capture sensor trends and variation during engine degradation.

## LSTM Sequence Generation

The LSTM uses a sliding window of 30 cycles.

Example:

Cycle 1 → Cycle 30 → RUL at Cycle 30
Cycle 2 → Cycle 31 → RUL at Cycle 31
Cycle 3 → Cycle 32 → RUL at Cycle 32

Sequences are created separately for each engine to prevent sequences from crossing between different engine IDs.

## Model Comparison

| Model | Input | Result |
|---|---|---|
| XGBoost | Engineered sensor features | Better |
| LSTM | 30-cycle sequences | Lower performance |

XGBoost currently performs better than the lightweight LSTM on the validation dataset.

### Current Results

- XGBoost Validation RMSE: Add actual value
- XGBoost Validation MAE: Add actual value
- LSTM Validation RMSE: Add actual value
- Failure threshold: RUL < 20 cycles

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
│   │   ├── xgboost_model.py
│   │   ├── xgboost_rul_model.json
│   │   ├── lstm_model.py
│   │   └── lstm_rul_model.pth
│   └── rca/

├── requirements.txt
└── README.md

## Failure Prediction

A simple threshold is used to identify engines that may fail soon:

Predicted RUL < 20 cycles
        ↓
Failing Soon = True

This converts the continuous RUL prediction into a simple maintenance-risk indicator.

## Next Steps

- [x] LSTM RUL prediction
- [x] Model comparison
- [ ] SHAP explainability
- [ ] Root Cause Analysis
- [ ] Maintenance recommendations
- [ ] FastAPI
- [ ] Streamlit dashboard
- [ ] MLflow
- [ ] Docker

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · PyTorch · SHAP · FastAPI · Streamlit · PostgreSQL · MLflow · Docker