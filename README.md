# Internals_Basics
# 🚀 TokenCost MLOps Pipeline

This repository contains a complete end-to-end MLOps pipeline developed as part of a CIE assignment. The objective is to predict **tokens_per_second** for LLM inference systems and implement a full model lifecycle.

---

## 📊 Problem Statement

Given system-level features:
- `model_params_billions`
- `prompt_tokens`
- `batch_size`
- `gpu_memory_gb`

Predict:
- `tokens_per_second`

---

## 🧠 Pipeline Overview

Data → Training → MLflow Tracking → Hyperparameter Tuning → Docker Deployment → Retraining

---

## 📁 Project Structure
MLOPS_Lab_CIE/
│
├── data/
│ ├── training_data.csv
│ └── new_data.csv
│
├── src/
│ ├── train.py
│ ├── tune.py
│ ├── predict_cli.py
│ └── retrain.py
│
├── results/
│ ├── step1_s1.json
│ ├── step2_s2.json
│ ├── step3_s3.json
│ └── step4_s8.json
│
├── Dockerfile
├── requirements.txt
└── README.md

---

## ⚙️ Tasks Implemented

### 🔹 Task 1 — Model Training & MLflow Tracking
- Trained **Lasso** and **GradientBoosting**
- Used train-test split (`test_size=0.2`, `random_state=42`)
- Logged:
  - Parameters
  - Metrics (MAE, RMSE, R², MAPE)
  - Tag: `priority = high`
- Selected best model using **RMSE**
- Output: `results/step1_s1.json`

---

### 🔹 Task 2 — Hyperparameter Tuning
- Performed **Random Search**
- Used **3-Fold Cross Validation**
- Tuned **GradientBoosting**
- Logged nested runs in MLflow
- Selected best configuration using RMSE
- Output: `results/step2_s2.json`

---

### 🔹 Task 3 — Docker Deployment
- Built CLI prediction tool using `argparse`
- Containerized using Docker
- Base image: `python:3.10-slim`

Run command:
docker run tokencost-predictor:v1 python src/predict_cli.py
--model_params_billions 46.6
--prompt_tokens 1730
--batch_size 11
--gpu_memory_gb 47

- Output: `results/step3_s3.json`

---

### 🔹 Task 4 — Retraining Pipeline
- Combined training + new data
- Retrained **Lasso model** (champion model)
- Compared retrained vs original model on same test set
- Promotion logic:
  - If RMSE improves ≥ 0.5 → promote
  - Else → keep existing model
- Output: `results/step4_s8.json`

---

## 📈 Key Results

- Best initial model: **Lasso**
- Tuned model: **GradientBoosting**
- Retraining improvement: **~8.32 RMSE reduction**
- Final decision: **Model Promoted**

---

## 🧠 Concepts Demonstrated

- MLflow Experiment Tracking
- Model Comparison & Evaluation
- Hyperparameter Tuning (Random Search)
- Cross Validation
- Docker Containerization
- CLI-based Inference
- Model Retraining Pipeline
- Model Promotion Strategy

---

## 🎯 Conclusion

This project implements a complete MLOps workflow:

**Training → Tracking → Tuning → Deployment → Monitoring → Retraining**

It demonstrates practical understanding of production-level machine learning systems.

---