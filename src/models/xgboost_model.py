import pandas as pd
import mlflow
import mlflow.xgboost

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


TRAIN_PATH = "data/processed/train_processed.csv"
VAL_PATH = "data/processed/val_processed.csv"
MODEL_PATH = "src/models/xgboost_rul_model.json"


# Load data

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)


# Input and target

drop_col = ["unit", "RUL"]

x_train = train_df.drop(columns=drop_col)
y_train = train_df["RUL"]

x_val = val_df.drop(columns=drop_col)
y_val = val_df["RUL"]


# Create parameter grid

param_grid = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5]
}


# Create XGBoost model

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# Randomized search

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# Start MLflow experiment

mlflow.set_experiment("Predictive-Maintenance")
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "Predictive-Maintenance"
)

with mlflow.start_run(run_name="XGBoost-RUL"):

    # Hyperparameter tuning

    search.fit(x_train, y_train)

    best_model = search.best_estimator_

    print("Best Parameters:")
    print(search.best_params_)

    print("Best CV RMSE:")
    print(-search.best_score_)


    # Prediction

    predicted_rul = best_model.predict(x_val)


    # Calculate errors

    rmse = mean_squared_error(
        y_val,
        predicted_rul
    ) ** 0.5

    mae = mean_absolute_error(
        y_val,
        predicted_rul
    )


    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))


    # Failure prediction

    val_df["predicted_RUL"] = predicted_rul

    val_df["failing_soon"] = (
        val_df["predicted_RUL"] < 20
    )


    # MLflow parameters

    mlflow.log_params(
        search.best_params_
    )

    mlflow.log_param(
        "model",
        "XGBRegressor"
    )

    mlflow.log_param(
        "cv",
        3
    )

    mlflow.log_param(
        "n_iter",
        20
    )

    mlflow.log_param(
        "failure_threshold",
        20
    )


    # MLflow metrics

    mlflow.log_metric(
        "cv_rmse",
        -search.best_score_
    )

    mlflow.log_metric(
        "validation_rmse",
        rmse
    )

    mlflow.log_metric(
        "validation_mae",
        mae
    )


    # Save model

    best_model.save_model(
        MODEL_PATH
    )


    # Log model to MLflow

    mlflow.xgboost.log_model(
        best_model,
        "xgboost_model"
    )


    # Sample prediction

    print("\nSample Predictions:")

    print(
        val_df[
            [
                "unit",
                "cycle",
                "RUL",
                "predicted_RUL",
                "failing_soon"
            ]
        ].head(10)
    )


print("\nXGBoost training completed.")