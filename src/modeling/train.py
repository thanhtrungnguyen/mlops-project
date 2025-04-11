from pathlib import Path
import sys

from loguru import logger
from tqdm import tqdm
import typer
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, log_loss, classification_report,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
import numpy as np
import time
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def perform_training(dataset_path, model_path):
    # Load data and split
    data = pd.read_csv(dataset_path)
    x = data.drop(columns=["target"])
    y = data["target"]
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)

    # Convert all integer columns to float64 in the entire dataset upfront
    for col in x_train.select_dtypes(include=['int', 'int64']).columns:
        x_train[col] = x_train[col].astype('float64')
        x_val[col] = x_val[col].astype('float64')
        x_test[col] = x_test[col].astype('float64')

    # Random Forest training with GridSearchCV
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20]
    }
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring="accuracy")

    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        # Track training time
        start_time = time.time()
        grid_search.fit(x_train, y_train)
        training_time = time.time() - start_time
        best_rf = grid_search.best_estimator_

        # Predictions
        y_pred = best_rf.predict(x_val)
        y_pred_proba = best_rf.predict_proba(x_val)

        # Basic metrics
        acc = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted")
        recall = recall_score(y_val, y_pred, average="weighted")
        f1 = f1_score(y_val, y_pred, average="weighted")

        # Additional metrics
        mcc = matthews_corrcoef(y_val, y_pred)
        kappa = cohen_kappa_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)

        # Advanced metrics
        conf_matrix = confusion_matrix(y_val, y_pred)
        roc_auc = None
        logloss = None

        # Combine all metrics into a single dictionary
        metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_time": training_time,
            "matthews_corrcoef": mcc,
            "cohen_kappa": kappa,
            "balanced_accuracy": balanced_acc
        }

        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
            metrics["roc_auc"] = roc_auc
        except:
            logger.warning("ROC AUC could not be calculated")

        try:
            logloss = log_loss(y_val, y_pred_proba)
            metrics["log_loss"] = logloss
        except:
            logger.warning("Log loss could not be calculated")

        # Log metrics and parameters to MLflow in one operation
        mlflow.log_metrics(metrics)
        mlflow.log_params(grid_search.best_params_)

        # Log CV results
        cv_metrics = {f"cv_fold_{i}_score": score for i, score in enumerate(grid_search.cv_results_['mean_test_score'])}
        mlflow.log_metrics(cv_metrics)

        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': x_train.columns,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Log all results to console in consolidated format
        metrics_str = "\n".join([f"- {k}: {v:.4f}" for k, v in metrics.items()])
        training_results_section = f"\n==== Training Results ====\nBest parameters: {grid_search.best_params_}\nTraining time: {training_time:.2f} seconds"
        metrics_section = f"\n==== Metrics ====\n{metrics_str}"
        confusion_matrix_section = f"\n==== Confusion Matrix ====\n{conf_matrix}"
        feature_importance_section = f"\n==== Top 10 Feature Importance ====\n{feature_importance.head(10).to_string(index=False)}"
        classification_report_section = f"\n==== Classification Report ====\n{classification_report(y_val, y_pred)}"

        logger.info(f"{training_results_section}\n{metrics_section}\n{confusion_matrix_section}\n{feature_importance_section}\n{classification_report_section}")


@app.command()
def main(
        dataset_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
        model_path: Path = MODELS_DIR / "best_model",
):
    try:
        logger.info("Training some model...")
        perform_training(str(dataset_path), str(model_path))
        logger.success("Modeling training complete.")
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    app()
