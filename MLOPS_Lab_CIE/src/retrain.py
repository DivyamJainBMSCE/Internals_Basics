import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("../data/training_data.csv")
new_df = pd.read_csv("../data/new_data.csv")

combined_df = pd.concat([train_df, new_df], ignore_index=True)

# =========================
# SPLIT (SAME TEST SET LOGIC)
# =========================
X = train_df.drop("tokens_per_second", axis=1)
y = train_df["tokens_per_second"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# CHAMPION MODEL (Task 1 → Lasso)
# =========================
champion = Lasso(alpha=1.0, random_state=42)
champion.fit(X_train, y_train)
champion_preds = champion.predict(X_test)

champion_rmse = np.sqrt(mean_squared_error(y_test, champion_preds))

# =========================
# RETRAINED MODEL (ON COMBINED DATA)
# =========================
X_comb = combined_df.drop("tokens_per_second", axis=1)
y_comb = combined_df["tokens_per_second"]

retrained = Lasso(alpha=1.0, random_state=42)
retrained.fit(X_comb, y_comb)
retrained_preds = retrained.predict(X_test)

retrained_rmse = np.sqrt(mean_squared_error(y_test, retrained_preds))

# =========================
# IMPROVEMENT CHECK
# =========================
improvement = champion_rmse - retrained_rmse

threshold = 0.5

if improvement >= threshold:
    action = "promoted"
else:
    action = "kept_champion"

# =========================
# SAVE JSON
# =========================
output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": float(champion_rmse),
    "retrained_rmse": float(retrained_rmse),
    "improvement": float(improvement),
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("../results", exist_ok=True)

with open("../results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed")