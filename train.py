import argparse
import os
import pandas as pd
from sklearn.metrics import f1_score
import joblib
from xgboost import XGBClassifier
import mlflow

# Argumentos del script
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=10)
parser.add_argument('--max_depth', type=int, default=7)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--metrics_output', type=str)
parser.add_argument('--model_output', type=str)
args = parser.parse_args()

print('ARGS:')
print('\n')
print('n_estimators:', args.n_estimators)
print('max_depth:', args.max_depth)
print('data_dir:', args.data_dir)
print('metrics_output:', args.metrics_output)
print('model_output:', args.model_output)
print('\n')

print("✅ 1. Se recibieron bien los argumentos")

# Cargar datos
X_train = pd.read_csv(f"{args.data_dir}/X_train.csv")
y_train = pd.read_csv(f"{args.data_dir}/y_train.csv")
X_val = pd.read_csv(f"{args.data_dir}/X_val.csv")
y_val = pd.read_csv(f"{args.data_dir}/y_val.csv")

print("✅ 2. Se cargaron bien los datasets")

# Entrenar modelo XGBoost
model = XGBClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=42
)

print("✅ 3. Se definio bien el modelo")

with mlflow.start_run():

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    model.fit(X_train, y_train)

    print("✅ 4. Entrenamiento listo")

    # Train

    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_pred_prob >= 0.25)

    f1_train = f1_score(y_train, y_train_pred, average = 'macro')
    print('f1 train:', round(f1_train*100, 2))

    # Val

    y_val_pred_prob = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_pred_prob >= 0.25)

    f1 = f1_score(y_val, y_val_pred, average = 'macro')
    print('f1_score:', round(f1*100, 2))

    mlflow.log_metric("f1_score", f1)

    print("✅ 5. Evaluación lista")

    # Guardar modelo como artefacto
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.pkl")
    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print("✅ 6. Modelo guardado")

    # Guardar métrica como archivo
    os.makedirs(args.metrics_output, exist_ok=True)
    metrics_path = os.path.join(args.metrics_output, "mlflow_metrics.json")
    with open(metrics_path, "w") as f:
        f.write(f'{{"f1_score": {f1}}}')
    mlflow.log_artifact(metrics_path, artifact_path="metrics")

    print("✅ 7. Metricas guardadas")