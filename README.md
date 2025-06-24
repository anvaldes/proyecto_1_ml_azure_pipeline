# 🤖 XGBoost Training + Evaluation Pipeline on Azure ML (Docker + Custom Metrics)

This repository contains a minimal yet powerful setup to run an **ML pipeline on Azure Machine Learning**, composed of:

- A training step using **XGBoost**
- A separate evaluation step
- Both steps executed using a **custom Docker image**

The main objective is to **train a classifier** and compare the **F1-score on validation and test datasets**, all orchestrated with the Azure ML SDK v2.

---

## 🚀 Features

- Uses a **custom Docker environment** defined via `Dockerfile`
- Trains an **XGBoost** model with `train.py`
- Computes `f1_score` on validation and test sets
- Saves:
  - The trained model (`model.pkl`)
  - Validation metrics in a JSON file (`mlflow_metrics.json`)
- Evaluation step (`evaluate.py`) compares test vs. validation performance
- Fully compatible with Azure ML Pipelines (SDK v2)

---

## ⚙️ How It Works

### 1. `train.py`

- Reads CLI arguments:
  - `--data_dir`: folder containing training/validation CSVs
  - `--n_estimators`, `--max_depth`: hyperparameters (optional)
  - `--model_output`, `--metrics_output`: output folders
- Trains an `XGBClassifier`
- Logs:
  - `f1_score` on validation set using `mlflow`
- Saves:
  - `model.pkl` to `model_output/`
  - `mlflow_metrics.json` with validation score to `metrics_output/`

---

### 2. `evaluate.py`

- Receives:
  - `--model_dir`: trained model folder from previous step
  - `--data_dir`: contains test set
  - `--metrics_dir`: contains validation score JSON
- Loads model and test data
- Computes F1 on test set
- Compares it against validation F1 from the previous step

---

### 3. `run_pipeline.py`

This script:

- Connects to your Azure ML workspace
- Builds the custom environment from the local `Dockerfile`
- Defines the pipeline with two steps:
  1. `train_step`
  2. `eval_step`
- Connects the outputs of the training step (`model`, `metrics`) to the evaluation step
- Launches the job and prints the experiment tracking URL

Run the pipeline with:

```bash
python run_pipeline.py
```

---
