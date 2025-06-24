import argparse
import os
import pandas as pd
import joblib
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)
parser.add_argument('--data_dir', type=str)
args = parser.parse_args()

print('ARGS:')
print('\n')
print('model_dir:', args.model_dir)
print('data_dir:', args.data_dir)
print('\n')

print("📦 Evaluando modelo...")

# Load model
model = joblib.load(os.path.join(args.model_dir, "model.pkl"))

# Load test data
X_test = pd.read_csv(os.path.join(args.data_dir, "X_val.csv"))
y_test = pd.read_csv(os.path.join(args.data_dir, "y_val.csv"))

# Evaluate
y_test_pred_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_prob >= 0.25)

f1_test = f1_score(y_test, y_test_pred, average = 'macro')
print('f1_score:', round(f1_test*100, 2))
