import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODEL_NAME = "Telco_Churn_Production_Model"

def train_and_register_model():
    print("[INFO] Starting Lab 6 MLflow Model Registry...")

    mlflow.set_experiment("Telco_Churn_Prediction")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42,
        "class_weight": "balanced",
    }

    with mlflow.start_run(run_name="RandomForest_Registry_V2") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")
        mlflow.log_param("dataset", "ChurnPrediction Telco Customer Churn")
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        if os.path.exists("models/preprocessor.pkl"):
            mlflow.log_artifact(
                "models/preprocessor.pkl",
                artifact_path="preprocessing_pipeline"
            )

        print("[INFO] Registering RandomForest model...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            registered_model_name=MODEL_NAME,
        )

        print(
            f"[SUCCESS] Registered {MODEL_NAME} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

if __name__ == "__main__":
    train_and_register_model()
