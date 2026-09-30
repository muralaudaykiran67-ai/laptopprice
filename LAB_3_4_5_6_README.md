# ChurnPrediction Labs 3-6

## Data rule
All labs use the **ChurnPrediction** raw Telco Customer Churn CSV in `data/raw/`. The dataset from `malli.zip` is used only as a reference for lab structure/code patterns; it is not copied into this project.

## Lab 3 - Baseline ML Pipeline
Run:
```powershell
python pipelines/run_lab3_baseline.py
```
Steps: preprocessing -> Random Forest baseline -> evaluation/error analysis.

## Lab 4 - MLflow Experiment Tracking
Run after installing requirements:
```powershell
python pipelines/run_lab4_tracking.py
```
Steps: deterministic preprocessing -> MLflow tracking -> reproducibility validation.

## Lab 5 - Production Preprocessing Pipeline
```powershell
python pipelines/run_lab5_pipeline.py
```
Uses sklearn Pipeline + ColumnTransformer and validates outputs.

## Lab 6 - Model Registry & Lifecycle
```powershell
python pipelines/run_lab6_registry.py
```
Uses the Lab 5 processed data for model registry/lifecycle automation.

## Important
Do not replace the ChurnPrediction raw CSV with the CSV from `malli.zip`.
