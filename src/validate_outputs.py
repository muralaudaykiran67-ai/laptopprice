import os
import json
import numpy as np

def validate_preprocessing_outputs():
    print("[INFO] Validating Lab 5 preprocessing outputs...")

    required = [
        "data/processed/X_train_final.npy",
        "data/processed/X_test_final.npy",
        "data/processed/y_train.npy",
        "data/processed/y_test.npy",
        "models/preprocessor.pkl",
        "data/processed/dataset_metadata.json",
    ]

    missing = [p for p in required if not os.path.exists(p)]
    errors = [f"Missing output: {p}" for p in missing]

    if not errors:
        X_train = np.load(required[0])
        X_test = np.load(required[1])
        y_train = np.load(required[2])
        y_test = np.load(required[3])

        if np.isnan(X_train).sum() > 0:
            errors.append("NaNs detected in X_train.")
        if np.isnan(X_test).sum() > 0:
            errors.append("NaNs detected in X_test.")
        if X_train.shape[1] != X_test.shape[1]:
            errors.append("X_train and X_test feature dimensions differ.")
        if X_train.shape[0] != y_train.shape[0]:
            errors.append("X_train/y_train row mismatch.")
        if X_test.shape[0] != y_test.shape[0]:
            errors.append("X_test/y_test row mismatch.")

        with open("data/processed/dataset_metadata.json") as f:
            metadata = json.load(f)

        if metadata.get("source_rows") != 7043:
            errors.append(
                f"Expected the ChurnPrediction dataset to contain 7043 rows; "
                f"metadata reports {metadata.get('source_rows')}."
            )

        report = {
            "validation_status": "PASSED" if not errors else "FAILED",
            "matrix_dimensions": {
                "X_train_shape": list(X_train.shape),
                "X_test_shape": list(X_test.shape),
                "y_train_shape": list(y_train.shape),
                "y_test_shape": list(y_test.shape),
            },
            "data_quality": {
                "missing_values_X_train": int(np.isnan(X_train).sum()),
                "missing_values_X_test": int(np.isnan(X_test).sum()),
            },
            "errors": errors,
        }
    else:
        report = {"validation_status": "FAILED", "errors": errors}

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/preprocessing_summary_report.json", "w") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output validation FAILED!")
        for e in errors:
            print(" -", e)
        return False

    print("[SUCCESS] Output validation PASSED.")
    print("[INFO] Processed data contains no NaNs and train/test dimensions match.")
    return True

if __name__ == "__main__":
    if not validate_preprocessing_outputs():
        raise SystemExit(1)
