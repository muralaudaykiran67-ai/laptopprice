import os
import pandas as pd

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

ALLOWED = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    },
    "Churn": {"Yes", "No"},
}

def locate_raw_dataset():
    raw_dir = "data/raw"
    candidates = [
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv")
    ] if os.path.isdir(raw_dir) else []
    if not candidates:
        raise FileNotFoundError("No CSV file found in data/raw/")
    # Prefer the existing Telco file if present; otherwise use the only CSV.
    preferred = [p for p in candidates if "telco" in os.path.basename(p).lower()]
    return preferred[0] if preferred else candidates[0]

def validate_schema(df):
    errors = []

    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(
            f"Column mismatch. Expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
        )

    if len(df) == 0:
        errors.append("Dataset is empty.")

    if df["customerID"].isna().any():
        errors.append("customerID contains missing values.")

    for col, allowed in ALLOWED.items():
        if col not in df.columns:
            continue
        observed = set(df[col].dropna().astype(str).str.strip().unique())
        bad = observed - {str(x) for x in allowed}
        if bad:
            errors.append(f"{col} contains unexpected values: {sorted(bad)}")

    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges"]
    for col in numeric:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            errors.append(f"{col} contains non-numeric values.")
        elif (converted < 0).any():
            errors.append(f"{col} contains negative values.")

    # TotalCharges is allowed to contain blank strings in the raw Telco dataset.
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if total.dropna().lt(0).any():
        errors.append("TotalCharges contains negative values.")

    return errors

def main():
    path = locate_raw_dataset()
    df = pd.read_csv(path)
    print(f"[INFO] Validating: {path}")
    print(f"[INFO] Records: {len(df)} | Columns: {len(df.columns)}")

    errors = validate_schema(df)

    os.makedirs("artifacts", exist_ok=True)
    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "dataset_path": path,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "errors": errors,
    }
    with open("artifacts/baseline_validation.json", "w") as f:
        import json
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Schema validation FAILED:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    print("[SUCCESS] Schema validation PASSED.")
    print("[INFO] ChurnPrediction dataset remains unchanged.")

if __name__ == "__main__":
    main()
