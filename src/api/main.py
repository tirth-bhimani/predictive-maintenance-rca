from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

app = FastAPI(title="Predictive Maintenance API")

MODEL_PATH = "src/models/xgboost_rul_model.json"

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

explainer = shap.TreeExplainer(model)


class PredictionRequest(BaseModel):
    unit: int
    cycle: float

    setting1: float
    setting2: float
    setting3: float

    sensor1: float
    sensor2: float
    sensor3: float
    sensor4: float
    sensor5: float
    sensor6: float
    sensor7: float
    sensor8: float
    sensor9: float
    sensor10: float
    sensor11: float
    sensor12: float
    sensor13: float
    sensor14: float
    sensor15: float
    sensor16: float
    sensor17: float
    sensor18: float
    sensor19: float
    sensor20: float
    sensor21: float

    sensor1_mean_10: float
    sensor1_std_10: float
    sensor2_mean_10: float
    sensor2_std_10: float
    sensor3_mean_10: float
    sensor3_std_10: float
    sensor4_mean_10: float
    sensor4_std_10: float
    sensor5_mean_10: float
    sensor5_std_10: float
    sensor6_mean_10: float
    sensor6_std_10: float
    sensor7_mean_10: float
    sensor7_std_10: float
    sensor8_mean_10: float
    sensor8_std_10: float
    sensor9_mean_10: float
    sensor9_std_10: float
    sensor10_mean_10: float
    sensor10_std_10: float
    sensor11_mean_10: float
    sensor11_std_10: float
    sensor12_mean_10: float
    sensor12_std_10: float
    sensor13_mean_10: float
    sensor13_std_10: float
    sensor14_mean_10: float
    sensor14_std_10: float
    sensor15_mean_10: float
    sensor15_std_10: float
    sensor16_mean_10: float
    sensor16_std_10: float
    sensor17_mean_10: float
    sensor17_std_10: float
    sensor18_mean_10: float
    sensor18_std_10: float
    sensor19_mean_10: float
    sensor19_std_10: float
    sensor20_mean_10: float
    sensor20_std_10: float
    sensor21_mean_10: float
    sensor21_std_10: float


@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    data = request.model_dump()

    unit = data.pop("unit")

    X = pd.DataFrame([data])

    feature_names = model.get_booster().feature_names

    X = X[feature_names]

    predicted_rul = float(model.predict(X)[0])

    if predicted_rul < 20:
        risk = "HIGH"
    elif predicted_rul < 50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    shap_values = explainer.shap_values(X)

    importance = np.abs(shap_values[0])

    factors = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    factors = factors.sort_values(
        "importance",
        ascending=False
    ).head(3)

    top_factors = factors.to_dict(
        orient="records"
    )

    return {
        "unit": unit,
        "predicted_rul": round(predicted_rul, 2),
        "risk": risk,
        "top_factors": top_factors
    }