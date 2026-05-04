import pandas as pd
import numpy as np
import json
import os
import random

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/training_data.csv")

X = df.drop("tokens_per_second", axis=1)
y = df["tokens_per_second"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# PARAM GRID
# =========================
param_grid = {
    "n_estimators": [50, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

# Create all combinations
all_combinations = [
    (n, lr, d)
    for n in param_grid["n_estimators"]
    for lr in param_grid["learning_rate"]
    for d in param_grid["max_depth"]
]

# Random search: shuffle and take subset
random.shuffle(all_combinations)
n_trials = 5
trials = all_combinations[:n_trials]


# =========================
# METRIC FUNCTION
# =========================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


# =========================
# MLFLOW PARENT RUN
# =========================
mlflow.set_experiment("tokencost-tokens-per-second")

best_rmse = float("inf")
best_params = None
best_mae = None
best_cv_mae = None

kf = KFold(n_splits=3, shuffle=True, random_state=42)

with mlflow.start_run(run_name="tuning-tokencost"):

    for i, (n, lr, d) in enumerate(trials):

        with mlflow.start_run(run_name=f"trial_{i}", nested=True):

            model = GradientBoostingRegressor(
                n_estimators=n,
                learning_rate=lr,
                max_depth=d,
                random_state=42
            )

            cv_mae_scores = []
            cv_rmse_scores = []

            # 3-fold CV
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)

                mae, rmse = compute_metrics(y_val, preds)
                cv_mae_scores.append(mae)
                cv_rmse_scores.append(rmse)

            avg_mae = np.mean(cv_mae_scores)
            avg_rmse = np.mean(cv_rmse_scores)

            # Log params
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("max_depth", d)

            # Log metrics
            mlflow.log_metric("cv_mae", avg_mae)
            mlflow.log_metric("cv_rmse", avg_rmse)

            # Track best
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = {
                    "n_estimators": n,
                    "learning_rate": lr,
                    "max_depth": d
                }
                best_mae = avg_mae
                best_cv_mae = avg_mae


# =========================
# SAVE JSON
# =========================
output = {
    "search_type": "random",
    "n_folds": 3,
    "total_trials": n_trials,
    "best_params": best_params,
    "best_mae": float(best_mae),
    "best_cv_mae": float(best_cv_mae),
    "parent_run_name": "tuning-tokencost"
}

os.makedirs("../results", exist_ok=True)

with open("../results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 completed successfully")