import random

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse
)

from src.rca.root_cause import (
    load_model,
    get_root_cause
)


# -----------------------------------
# Settings
# -----------------------------------

FAILURE_THRESHOLD = 20


# -----------------------------------
# Create FastAPI application
# -----------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="RUL Prediction and Root Cause Analysis",
    version="1.0"
)


# -----------------------------------
# Load XGBoost model
# -----------------------------------

model = load_model()


# -----------------------------------
# Get feature names
# -----------------------------------

feature_names = (
    model.get_booster().feature_names
)


# -----------------------------------
# Home endpoint
# -----------------------------------

@app.get("/")
def home():

    return {
        "message": "Predictive Maintenance API is running"
    }


# -----------------------------------
# Prediction endpoint
# -----------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):

    # Convert input features

    features = np.array(
        request.features,
        dtype=float
    )


    # Check number of features

    if len(features) != len(feature_names):

        raise HTTPException(

            status_code=400,

            detail={
                "message": "Incorrect number of features",

                "expected_features": len(feature_names),

                "received_features": len(features)
            }
        )


    # Create DataFrame

    X = pd.DataFrame(
        [features],
        columns=feature_names
    )


    # Predict RUL

    predicted_rul = float(
        model.predict(X)[0]
    )


    # Failure prediction

    failing_soon = (
        predicted_rul < FAILURE_THRESHOLD
    )


    # Root Cause Analysis

    rca_result = get_root_cause(

        model=model,

        X=X,

        feature_names=feature_names,

        rul=predicted_rul,

        top_n=3
    )


    # Return final response

    return {

        "RUL": round(
            predicted_rul,
            2
        ),

        "failing_soon": failing_soon,

        "top_factors": rca_result[
            "top_factors"
        ],

        "root_cause": rca_result[
            "root_cause"
        ],

        "recommendation": rca_result[
            "recommendation"
        ]
    }
    
    
    
    #-----------------------------------
    # Testing
    #-----------------------------------
# @app.get("/test-input")
# def test_input():

#     sample = pd.read_csv(
#         "data/processed/val_processed.csv"
#     )

#     sample = sample.drop(
#         columns=["unit", "RUL"]
#     )

#     return {
#         "features": sample.iloc[random.randint(0, len(sample))].tolist()
#     }