"""
modelling.py untuk MLflow Project
- Menerima hyperparameter via argparse
- Menjalankan training
- Logging ke MLflow (lokal, nanti di workflow bisa upload ke DagsHub atau lokal)
- Menyimpan model dan artefak
"""
import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=str, default="None")
    args = parser.parse_args()

    # Konversi max_depth
    max_depth = None if args.max_depth == "None" else int(args.max_depth)

    # Load data
    df = pd.read_csv("seattle-weather_preprocessing.csv")
    X = df[["precipitation", "temp_max", "temp_min", "wind"]]
    y = df["weather"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Simpan model
        mlflow.sklearn.log_model(model, "model")

        # Simpan classification report sebagai artefak
        report = classification_report(y_test, y_pred, output_dict=True)
        with open("report.json", "w") as f:
            json.dump(report, f)
        mlflow.log_artifact("report.json")

        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()