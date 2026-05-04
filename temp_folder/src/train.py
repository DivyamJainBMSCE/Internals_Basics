import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../data/training_data.csv")

X = df.drop("tokens_per_second", axis=1)
y = df["tokens_per_second"]


# =========================
# 2. TRAIN-TEST SPLIT (MANDATORY)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 3. METRIC FUNCTION
# =========================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Avoid divide-by-zero in MAPE
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    return mae, rmse, r2, mape


# =========================
# 4. SET MLFLOW EXPERIMENT
# =========================
mlflow.set_experiment("tokencost-tokens-per-second")


# =========================
# 5. DEFINE MODELS
# =========================
models = {
    "Lasso": Lasso(alpha=1.0, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


results = []
best_rmse = float("inf")
best_model_name = None


# =========================
# 6. TRAIN + LOG EACH MODEL
# =========================
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        
        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Metrics
        mae, rmse, r2, mape = compute_metrics(y_test, preds)
        
        # -------------------------
        # LOG PARAMETERS (REQUIRED)
        # -------------------------
        if name == "Lasso":
            mlflow.log_param("alpha", model.alpha)
        else:
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)
            mlflow.log_param("max_depth", model.max_depth)
        
        mlflow.log_param("model_name", name)
        
        # -------------------------
        # LOG METRICS (REQUIRED)
        # -------------------------
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)
        
        # -------------------------
        # TAG (REQUIRED)
        # -------------------------
        mlflow.set_tag("priority", "high")
        
        # Optional (good practice)
        mlflow.sklearn.log_model(model, "model")
        
        # Store result
        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape)
        })
        
        # Track best
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name


# =========================
# 7. SAVE JSON OUTPUT (CRITICAL)
# =========================
output = {
    "experiment_name": "tokencost-tokens-per-second",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": float(best_rmse)
}

os.makedirs("../results", exist_ok=True)

with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)


print("Task 1 completed successfully")
print("Best model:", best_model_name)