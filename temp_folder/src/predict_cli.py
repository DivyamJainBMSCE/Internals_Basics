import argparse
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

# Load training data
df = pd.read_csv("data/training_data.csv")

X = df.drop("tokens_per_second", axis=1)
y = df["tokens_per_second"]

# Tuned model from Task 2
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.2,
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# CLI arguments
parser = argparse.ArgumentParser()

parser.add_argument("--model_params_billions", type=float, required=True)
parser.add_argument("--prompt_tokens", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--gpu_memory_gb", type=int, required=True)

args = parser.parse_args()

# Prepare input
input_df = pd.DataFrame([{
    "model_params_billions": args.model_params_billions,
    "prompt_tokens": args.prompt_tokens,
    "batch_size": args.batch_size,
    "gpu_memory_gb": args.gpu_memory_gb
}])

# Predict
prediction = model.predict(input_df)[0]

print(float(prediction))