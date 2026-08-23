import numpy as np
import shap
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import xgboost as xgb

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide"
)

st.title("Predictive Maintenance Dashboard")


@st.cache_data
def load_data():
    return pd.read_csv(
        "D:\\resume\\ml_mini\\data\\processed\\train_processed.csv"
    )


@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model(
        "D:\\resume\\ml_mini\\src\\models\\xgboost_rul_model.json"
    )
    return model


df = load_data()
model = load_model()

units = st.selectbox(
    "Select Machine",
    sorted(df["unit"].unique())
)

unit_data = df[df["unit"] == units]

latest_cycle = unit_data["cycle"].max()

latest_data = unit_data[
    unit_data["cycle"] == latest_cycle
]

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted RUL")

    feature_columns = model.get_booster().feature_names

    X = latest_data[feature_columns]

    rul = float(model.predict(X)[0])

    st.metric(
        "Remaining Useful Life",
        f"{rul:.2f} cycles"
    )


with col2:
    st.subheader("Failure Risk")

    if rul < 20:
        st.error("HIGH RISK")
    elif rul < 50:
        st.warning("MEDIUM RISK")
    else:
        st.success("LOW RISK")


st.divider()

st.subheader("Top Root Cause Factors")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

importance = np.abs(shap_values[0])

factors = pd.DataFrame({
    "Factor": feature_columns,
    "Importance": importance
})

factors = factors.sort_values(
    "Importance",
    ascending=False
).head(3)

fig = px.bar(
    factors,
    x="Importance",
    y="Factor",
    orientation="h",
    title="Top 3 Factors"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.divider()

st.subheader("Current Alerts")

try:
    response = requests.get(
        f"{API_URL}/alerts",
        timeout=5
    )

    if response.status_code == 200:
        alerts = response.json()

        if alerts:
            for alert in alerts:
                st.error(
                    f"Unit {alert['unit']} - "
                    f"RUL: {alert['predicted_rul']} cycles - "
                    f"{alert['message']}"
                )
        else:
            st.success("No active alerts")

except requests.RequestException:
    st.info("Alert API is not available.")