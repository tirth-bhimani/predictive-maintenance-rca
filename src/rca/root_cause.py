import numpy as np
import shap
import xgboost as xgb


def load_model():

    model = xgb.XGBRegressor()

    model.load_model(
        "src/models/xgboost_rul_model.json"
    )

    return model


def get_root_cause(
    model,
    X,
    feature_names,
    top_n=3
):

    # Create SHAP explainer
    explainer = shap.TreeExplainer(
        model
    )

    # Calculate SHAP values
    shap_values = explainer.shap_values(
        X
    )

    # Get SHAP values for first prediction
    values = shap_values[0]

    # Create feature + SHAP value pairs
    results = []

    for feature, value in zip(
        feature_names,
        values
    ):

        if value < 0:

            impact = "decreases RUL"

        else:

            impact = "increases RUL"

        results.append({
            "feature": feature,
            "shap_value": float(value),
            "impact": impact,
            "importance": abs(float(value))
        })

    # Sort by absolute SHAP value
    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    # Return top features
    return results[:top_n]