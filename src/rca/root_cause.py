import shap
import xgboost as xgb

from src.rca.recommendation import (
    explain_feature,
    get_recommendation
)


MODEL_PATH = "src/models/xgboost_rul_model.json"


def load_model():

    model = xgb.XGBRegressor()

    model.load_model(MODEL_PATH)

    return model


def get_top_factors(
    model,
    X,
    feature_names,
    top_n=3
):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    values = shap_values[0]

    results = []

    for feature, value in zip(
        feature_names,
        values
    ):

        if value < 0:
            impact = "decreases RUL"
        else:
            impact = "increases RUL"

        feature_info = explain_feature(feature)

        results.append({
            "feature": feature,
            "shap_value": round(float(value), 4),
            "impact": impact,
            "importance": abs(float(value)),
            "sensor": feature_info["sensor"],
            "meaning": feature_info["meaning"],
            "component": feature_info["component"]
        })

    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return results[:top_n]


def analyze_root_cause(top_factors):

    component_scores = {}

    component_evidence = {}

    for factor in top_factors:

        component = factor["component"]

        if component == "Unknown":
            continue

        importance = factor["importance"]

        component_scores[component] = (
            component_scores.get(component, 0)
            + importance
        )

        if component not in component_evidence:
            component_evidence[component] = []

        component_evidence[component].append(
            factor
        )

    if not component_scores:

        return {
            "component": "Unknown",
            "fault": "Unable to determine likely component",
            "confidence": "low",
            "evidence": []
        }

    dominant_component = max(
        component_scores,
        key=component_scores.get
    )

    evidence = component_evidence[
        dominant_component
    ]

    if len(evidence) >= 2:
        confidence = "high"
    else:
        confidence = "medium"

    return {
        "component": dominant_component,
        "fault": f"Possible {dominant_component} degradation",
        "confidence": confidence,
        "evidence": evidence
    }


def get_root_cause(
    model,
    X,
    feature_names,
    rul,
    top_n=3
):

    top_factors = get_top_factors(
        model,
        X,
        feature_names,
        top_n
    )

    root_cause = analyze_root_cause(
        top_factors
    )

    recommendation = get_recommendation(
        rul,
        root_cause
    )

    return {
        "top_factors": top_factors,
        "root_cause": root_cause,
        "recommendation": recommendation
    }