import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_and_track(run_name="RandomForest_Baseline", params=None):
    if params is None:
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced"}
    X_train=np.load("data/processed/X_train_final.npy"); X_test=np.load("data/processed/X_test_final.npy")
    y_train=np.load("data/processed/y_train.npy"); y_test=np.load("data/processed/y_test.npy")
    mlflow.set_experiment("Telco_Churn_Prediction")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params); mlflow.log_param("model_family", "RandomForest")
        model=RandomForestClassifier(**params); model.fit(X_train,y_train)
        y_pred=model.predict(X_test); y_prob=model.predict_proba(X_test)[:,1]
        metrics={"accuracy":accuracy_score(y_test,y_pred),"precision":precision_score(y_test,y_pred),"recall":recall_score(y_test,y_pred),"f1_score":f1_score(y_test,y_pred),"roc_auc":roc_auc_score(y_test,y_prob)}
        mlflow.log_metrics(metrics)
        if os.path.exists("data/processed/dataset_metadata.json"): mlflow.log_artifact("data/processed/dataset_metadata.json", artifact_path="metadata")
        mlflow.sklearn.log_model(sk_model=model, name="model")
        joblib.dump(model,"models/random_forest_model.pkl")
        print(f"Tracked {run_name}: F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    train_and_track()
