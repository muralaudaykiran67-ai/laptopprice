import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "Telco_Churn_Production_Model"
METRIC = "recall"

def automate_champion_challenger():
    print("[INFO] Starting automated model lifecycle manager...")
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")

        new_models = [mv for mv in versions if mv.current_stage == "None"]
        for mv in new_models:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"[INFO] Version {mv.version} -> Staging")

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        staging = [mv for mv in versions if mv.current_stage == "Staging"]

        if not staging:
            print("[INFO] No Staging model found.")
            return

        best = None
        best_score = -1.0

        for mv in staging:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(METRIC, 0.0)
            print(f"[INFO] Candidate V{mv.version}: {METRIC}={score:.4f}")
            if score > best_score:
                best_score = score
                best = mv

        production = [
            mv for mv in versions if mv.current_stage == "Production"
        ]
        champion = production[0] if production else None

        promote = False
        if champion is None:
            promote = True
            print("[INFO] No current Production model; promoting challenger.")
        else:
            champ_run = client.get_run(champion.run_id)
            champ_score = champ_run.data.metrics.get(METRIC, 0.0)
            print(
                f"[INFO] Current champion V{champion.version}: "
                f"{METRIC}={champ_score:.4f}"
            )
            promote = best_score > champ_score

        if promote:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=best.version,
                stage="Production",
                archive_existing_versions=False,
            )
            print(f"[SUCCESS] Version {best.version} promoted to Production.")

            if champion:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=champion.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                print(f"[INFO] Version {champion.version} archived.")
        else:
            print("[INFO] Champion retained.")

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        for mv in versions:
            if mv.current_stage == "Staging":
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

        print("[SUCCESS] Model lifecycle completed.")

    except Exception as e:
        print(f"[ERROR] Lifecycle failed: {e}")
        raise

if __name__ == "__main__":
    automate_champion_challenger()
