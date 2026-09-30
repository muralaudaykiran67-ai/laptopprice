# Lab 5 and Lab 6 — ChurnPrediction

These labs are adapted from the workflow in `malli.zip`, but they use **only the dataset already present in this ChurnPrediction project**.

## Dataset rule

Do not replace the ChurnPrediction dataset with the dataset from malli.zip.

The project uses:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn (1).csv`

Expected source rows: **7043**

Target column: `Churn`

The preprocessing uses the same train/test settings as the existing project:

- test size = 0.20
- random state = 42
- stratified split
- `customerID` dropped
- `TotalCharges` converted to numeric and missing values handled as 0
- `Churn`: Yes = 1, No = 0
- numeric features scaled
- categorical features one-hot encoded

## Lab 5 — Production Data Pipeline

Run from the `ChurnPrediction` root:

```powershell
python pipelines/run_lab5_pipeline.py
```

It performs:

1. raw-data schema validation
2. sklearn preprocessing pipeline
3. output validation

Main outputs:

- `models/preprocessor.pkl`
- `data/processed/X_train_final.npy`
- `data/processed/X_test_final.npy`
- `data/processed/y_train.npy`
- `data/processed/y_test.npy`
- `data/processed/dataset_metadata.json`
- `artifacts/baseline_validation.json`
- `artifacts/preprocessing_summary_report.json`

## Lab 6 — Model Registry and Lifecycle

First complete Lab 5.

Then run:

```powershell
python pipelines/run_lab6_registry.py
```

It performs:

1. trains a Random Forest
2. logs parameters and metrics to MLflow
3. registers the model as `Telco_Churn_Production_Model`
4. moves new versions to Staging
5. compares Staging against the Production champion using Recall
6. promotes the challenger when its Recall is higher
7. archives the previous champion/losing Staging versions
8. creates `artifacts/production_model_report.json`

## MLflow UI

Start MLflow from the project root:

```powershell
mlflow ui
```

Then open the local MLflow URL shown in the terminal.

## Important

Lab 5 and Lab 6 are designed to be run from the `ChurnPrediction` root directory, not from inside `src` or `pipelines`.
