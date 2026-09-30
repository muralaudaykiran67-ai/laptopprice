import os
import json
from mlflow.tracking import MlflowClient

MODEL_NAME = "Telco_Churn_Production_Model"

def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    champion = next(
        (mv for mv in versions if mv.current_stage == "Production"),
        None,
    )

    if champion is None:
        raise RuntimeError("No model is currently in Production.")

    run = client.get_run(champion.run_id)

    report = {
        "registry_status": "READY_FOR_DEPLOYMENT",
        "dataset": "ChurnPrediction Telco Customer Churn",
        "source_dataset_rows": 7043,
        "model_lineage": {
            "registered_name": MODEL_NAME,
            "version": int(champion.version),
            "current_stage": champion.current_stage,
            "run_id": champion.run_id,
            "artifact_uri": champion.source,
        },
        "performance_metrics": {
            "recall": run.data.metrics.get("recall"),
            "f1_score": run.data.metrics.get("f1_score"),
            "roc_auc": run.data.metrics.get("roc_auc"),
            "accuracy": run.data.metrics.get("accuracy"),
            "precision": run.data.metrics.get("precision"),
        },
        "hyperparameters": run.data.params,
        "preprocessing_dependency": "models/preprocessor.pkl",
    }

    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/production_model_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Production report saved to {path}")

if __name__ == "__main__":
    generate_registry_report()
