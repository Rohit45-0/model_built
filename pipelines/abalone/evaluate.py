import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting evaluation.")

    # Define input/output paths
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_data_path = "/opt/ml/processing/test/test.csv"
    output_dir = "/opt/ml/processing/evaluation"

    # Extract model
    logger.info("Extracting model.")
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall(path=".")

    # Load XGBoost model
    logger.info("Loading model.")
    model = pickle.load(open("xgboost-model", "rb"))

    # Load test dataset
    logger.info("Reading test dataset.")
    df = pd.read_csv(test_data_path)  # Headers included from preprocess.py

    # Split features and labels
    y_test = df["Is Fraudulent"].to_numpy()  # Target column from preprocess.py
    X_test = df.drop(columns=["Is Fraudulent"])
    X_test_dmatrix = xgboost.DMatrix(X_test.values)

    # Perform predictions (probabilities)
    logger.info("Performing predictions.")
    probabilities = model.predict(X_test_dmatrix)
    predictions = (probabilities >= 0.5).astype(int)  # Threshold at 0.5

    # Compute evaluation metrics
    logger.info("Computing classification metrics.")
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)  # AUC uses probabilities

    # Create evaluation report
    report_dict = {
        "classification_metrics": {
            "f1": {"value": float(f1)},
            "precision": {"value": float(precision)},
            "recall": {"value": float(recall)},
            "auc": {"value": float(auc)}
        }
    }

    # Save evaluation results
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{output_dir}/evaluation.json"
    
    logger.info("Saving evaluation results to %s", evaluation_path)
    with open(evaluation_path, "w") as f:
        json.dump(report_dict, f)

    logger.info("Evaluation complete.")