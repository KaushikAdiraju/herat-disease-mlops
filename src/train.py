from src.preprocess import build_preprocessor, preprocess_and_save  # noqa: E402
import os
import sys
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import mlflow
import mlflow.sklearn

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
SCREENSHOT_DIR = os.path.join("reports", "screenshots")


def plot_evaluation_artifacts(y_true, y_pred, y_prob, model_name: str, output_dir: str):
    """Generates and saves Confusion Matrix and ROC Curve plots for MLflow & Reporting."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"cm_{safe_name}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#d95f02", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="#2b5c8f", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"roc_{safe_name}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    return cm_path, roc_path


def train_and_evaluate():
    """Trains multiple classifiers with GridSearchCV, logs to MLflow, and packages the best model."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # Ensure processed splits exist
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        logging.info("Processed splits not found. Running preprocessing first...")
        preprocess_and_save()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Configure MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Heart_Disease_Classification_MLOps")

    models_config = {
        "Logistic_Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear", "lbfgs"]
            }
        },
        "Random_Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5]
            }
        },
        "Gradient_Boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5]
            }
        }
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall_auc = -1.0
    best_overall_pipeline = None
    best_model_name = ""

    comparison_records = []

    for model_name, cfg in models_config.items():
        logging.info(f"--- Training & Tuning {model_name} ---")
        full_pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", cfg["estimator"])
        ])

        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=cfg["param_grid"],
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )

        with mlflow.start_run(run_name=model_name):
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            cv_best_auc = grid_search.best_score_

            # Evaluate on hold-out test set
            y_pred = best_pipeline.predict(X_test)
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "cv_mean_roc_auc": float(cv_best_auc),
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, y_prob))
            }

            # Log hyperparameters and metrics to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)

            # Generate and log plots as MLflow artifacts
            cm_path, roc_path = plot_evaluation_artifacts(
                y_test, y_pred, y_prob, model_name, SCREENSHOT_DIR
            )
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
            mlflow.log_artifact(roc_path, artifact_path="evaluation_plots")

            # Log the sklearn pipeline to MLflow
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path="model_pipeline",
                input_example=X_train.iloc[:2]
            )

            logging.info(f"{model_name} Best Params: {grid_search.best_params_}")
            logging.info(f"{model_name} Metrics: {metrics}")

            comparison_records.append({"model": model_name, **metrics})

            if metrics["test_roc_auc"] > best_overall_auc:
                best_overall_auc = metrics["test_roc_auc"]
                best_overall_pipeline = best_pipeline
                best_model_name = model_name

    # Save model comparison summary CSV for the final report
    comparison_df = pd.DataFrame(comparison_records)
    comparison_csv_path = os.path.join(SCREENSHOT_DIR, "model_comparison_metrics.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    logging.info(f"\nModel Comparison Summary:\n{comparison_df.to_string(index=False)}")

    # Save the winning end-to-end pipeline for FastAPI & Docker serving
    final_model_path = os.path.join(MODEL_DIR, "best_model_pipeline.joblib")
    joblib.dump(best_overall_pipeline, final_model_path)
    logging.info(
        f"Best overall model ({best_model_name} with ROC-AUC={best_overall_auc:.4f}) saved to {final_model_path}")

    return best_overall_pipeline, comparison_df


if __name__ == "__main__":
    train_and_evaluate()
