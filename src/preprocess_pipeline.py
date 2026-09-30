import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def locate_raw_dataset():
    raw_dir = "data/raw"
    candidates = [
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv")
    ]
    if not candidates:
        raise FileNotFoundError("No CSV file found in data/raw/")
    preferred = [p for p in candidates if "telco" in os.path.basename(p).lower()]
    return preferred[0] if preferred else candidates[0]

def run_preprocessing():
    print("[INFO] Starting Lab 5 Sklearn Production Pipeline...")

    data_path = locate_raw_dataset()
    df = pd.read_csv(data_path)

    # Keep the same ChurnPrediction dataset and preprocessing semantics.
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (
        df["Churn"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    )

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("ohe", OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore"
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    print("[INFO] Fitting preprocessing pipeline on training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    joblib.dump(preprocessor, "models/preprocessor.pkl")

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
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
        "preprocessor": "ColumnTransformer -> imputation/scaling + one-hot encoding",
    }

    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Lab 5 preprocessing completed.")
    print(f"[INFO] X_train shape: {X_train_final.shape}")
    print(f"[INFO] X_test shape : {X_test_final.shape}")
    print("[INFO] Saved models/preprocessor.pkl")

if __name__ == "__main__":
    run_preprocessing()
