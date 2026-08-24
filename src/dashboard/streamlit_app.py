import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://api:8000"

st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide"
)

st.title("Predictive Maintenance Dashboard")

st.info(
    "Demo mode: machine sensor data is simulated using "
    "the NASA C-MAPSS FD001 dataset."
)


@st.cache_data
def load_data():
    return pd.read_csv(
     "data/processed/train_processed.csv"
    )


df = load_data()

units = st.selectbox(
    "Select Machine",
    sorted(df["unit"].unique())
)

unit_data = df[df["unit"] == units]

max_cycle = int(unit_data["cycle"].max())

selected_cycle = st.slider(
    "Select Current Machine Cycle",
    min_value=1,
    max_value=max_cycle,
    value=max(1, max_cycle // 2)
)

cycle_data = unit_data[
    unit_data["cycle"] == selected_cycle
]

if cycle_data.empty:
    st.error("No data available for this cycle.")
    st.stop()

machine_data = cycle_data.iloc[0]

st.write(f"Machine: **{units}**")
st.write(f"Current Cycle: **{selected_cycle}**")

st.divider()

try:

    payload = machine_data.drop(
        labels=["RUL"],
        errors="ignore"
    ).to_dict()

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=10
    )

    if response.status_code != 200:

        st.error(
            f"Prediction failed: {response.text}"
        )

        st.stop()

    result = response.json()

    rul = result["predicted_rul"]
    risk = result["risk"]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Predicted RUL")

        st.metric(
            "Remaining Useful Life",
            f"{rul:.2f} cycles"
        )

    with col2:

        st.subheader("Failure Risk")

        if risk == "HIGH":

            st.error("HIGH RISK")

        elif risk == "MEDIUM":

            st.warning("MEDIUM RISK")

        else:

            st.success("LOW RISK")

    st.divider()

    st.subheader("Top Root Cause Factors")

    factors = pd.DataFrame(
        result["top_factors"]
    )

    factors = factors.rename(
        columns={
            "feature": "Factor",
            "importance": "Importance"
        }
    )

    fig = px.bar(
        factors,
        x="Importance",
        y="Factor",
        orientation="h",
        title="Top 3 Factors Affecting Prediction"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except requests.ConnectionError:

    st.error(
        "Cannot connect to FastAPI. "
        "Make sure the API is running."
    )

except requests.RequestException as e:

    st.error(
        f"API request failed: {e}"
    )