import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error

TRAIN_PATH = "data/processed/train_processed.csv"
VAL_PATH = "data/processed/val_processed.csv"

MODEL_PATH = "src/models/xgboost_rul_model.pkl"

#load data
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

#input and target
drop_col=["unit","RUL"]

x_train=train_df.drop(columns=drop_col)
y_train=train_df["RUL"]

x_val=val_df.drop(columns=drop_col)
y_val=val_df["RUL"]

#create model
param_grid = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5]
}


model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

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

search.fit(x_train, y_train)

best_model = search.best_estimator_

print("Best Parameters:")
print(search.best_params_)

print("Best CV RMSE:")
print(-search.best_score_)

#train model
model.fit(x_train,y_train)

#prediction
predicted_rul = best_model.predict(x_val)

#calculate error
rmse= mean_squared_error(
    y_val,
    predicted_rul
)**0.5

mae=mean_absolute_error(
    y_val,
    predicted_rul
)

print("rmse : ",round(rmse,2))
print("mae : ",round(mae,2))

#faliur prediction
val_df["predicted_RUL"] = predicted_rul

val_df["failing_soon"] = (
    val_df["predicted_RUL"] < 20
)

#save model
best_model.save_model(
    "src/models/xgboost_rul_model.json"
)

#sample prediction

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