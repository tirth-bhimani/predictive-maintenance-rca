import pandas as pd

from root_cause import (
    load_model,
    get_root_cause
)


# Load validation data

val_df = pd.read_csv(
    "data/processed/val_processed.csv"
)


# Prepare features

drop_col = [
    "unit",
    "RUL"
]

X = val_df.drop(
    columns=drop_col
)

feature_names = X.columns.tolist()


# Load XGBoost model

model = load_model()


# Select one prediction

sample = X.iloc[[0]]


# Predict RUL

prediction = model.predict(
    sample
)[0]

print(
    f"Predicted RUL: {prediction:.2f} cycles"
)


# Get root causes

root_causes = get_root_cause(
    model,
    sample,
    feature_names,
    top_n=3
)


# Display results

print("\nTop 3 Root Causes:")

for i, result in enumerate(
    root_causes,
    start=1
):

    print(
        f"{i}. {result['feature']}"
    )

    print(
        f"   SHAP value: "
        f"{result['shap_value']:.4f}"
    )

    print(
        f"   Impact: "
        f"{result['impact']}"
    )