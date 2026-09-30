import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def locate_raw_dataset():
    candidates = [os.path.join("data/raw", f) for f in os.listdir("data/raw") if f.lower().endswith(".csv")]
    if not candidates:
        raise FileNotFoundError("No CSV file found in data/raw/")
    return next((p for p in candidates if "telco" in os.path.basename(p).lower()), candidates[0])


def run_preprocessing():
    print("Starting Lab 3 preprocessing...")
    data_path = locate_raw_dataset()
    df = pd.read_csv(data_path)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().eq("yes").astype(int)

    X, y = df.drop("Churn", axis=1), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = StandardScaler()
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    X_train_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_scaled = scaler.transform(X_test[num_cols])
    X_train_encoded = ohe.fit_transform(X_train[cat_cols])
    X_test_encoded = ohe.transform(X_test[cat_cols])
    X_train_final = np.hstack((X_train_scaled, X_train_encoded))
    X_test_final = np.hstack((X_test_scaled, X_test_encoded))

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(ohe, "models/ohe.pkl")

    metadata = {
        "dataset_name": "Telco Customer Churn",
        "source_file": data_path,
        "source_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target": "Churn",
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True}
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Preprocessing completed successfully. Train: {X_train_final.shape}, Test: {X_test_final.shape}")

if __name__ == "__main__":
    run_preprocessing()
