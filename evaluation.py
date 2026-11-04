"""Evaluate the saved hybrid model and four classical baseline models."""

import logging
import os
from pathlib import Path

# Keep TensorFlow and parallel-library console messages concise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras


LOGGER = logging.getLogger("model_evaluation")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "processed_dataset.csv"
HYBRID_MODEL_PATH = PROJECT_ROOT / "models" / "best_hybrid_model.keras"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
TARGET_COLUMN = "stroke"
SPLIT_COLUMN = "data_split"
RANDOM_STATE = 42


def configure_logging() -> None:
    """Configure readable Phase 6 console logs."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sns.set_theme(style="whitegrid")


def load_evaluation_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the saved training and test partitions from Phase 3."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError("processed_dataset.csv was not found.")
    if not HYBRID_MODEL_PATH.is_file():
        raise FileNotFoundError("best_hybrid_model.keras was not found.")

    dataframe = pd.read_csv(DATASET_PATH)
    required_columns = {TARGET_COLUMN, SPLIT_COLUMN}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError("Processed data is missing stroke or data_split.")

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in required_columns
    ]
    training_data = dataframe[dataframe[SPLIT_COLUMN] == "train"]
    test_data = dataframe[dataframe[SPLIT_COLUMN] == "test"]
    if training_data.empty or test_data.empty:
        raise ValueError("Both training and test partitions are required.")

    x_train = training_data[feature_columns].to_numpy(dtype=np.float32)
    y_train = training_data[TARGET_COLUMN].to_numpy(dtype=np.int32)
    x_test = test_data[feature_columns].to_numpy(dtype=np.float32)
    y_test = test_data[TARGET_COLUMN].to_numpy(dtype=np.int32)
    LOGGER.info("Training records: %d", len(x_train))
    LOGGER.info("Test records: %d", len(x_test))
    LOGGER.info("Features: %d", len(feature_columns))
    return x_train, y_train, x_test, y_test


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate the five evaluation metrics requested in Phase 6."""
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1 Score": f1_score(actual, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(actual, probabilities),
    }


def evaluate_hybrid_model(
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Load and evaluate the saved CNN-RNN-LSTM checkpoint."""
    LOGGER.info("Loading saved hybrid model.")
    model = keras.models.load_model(HYBRID_MODEL_PATH)
    sequence_input = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    probabilities = model.predict(sequence_input, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int32)
    metrics = calculate_metrics(y_test, predictions, probabilities)
    return metrics, predictions, probabilities


def build_baseline_models() -> dict[str, object]:
    """Create the four simple comparison algorithms requested by the user."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_all_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Evaluate the hybrid model and train/evaluate all baselines."""
    metrics_rows: list[dict[str, object]] = []
    prediction_results: dict[str, dict[str, np.ndarray]] = {}

    hybrid_metrics, hybrid_predictions, hybrid_probabilities = (
        evaluate_hybrid_model(x_test, y_test)
    )
    metrics_rows.append({"Model": "CNN-RNN-LSTM", **hybrid_metrics})
    prediction_results["CNN-RNN-LSTM"] = {
        "predictions": hybrid_predictions,
        "probabilities": hybrid_probabilities,
    }

    for model_name, model in build_baseline_models().items():
        LOGGER.info("Training baseline: %s", model_name)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics_rows.append(
            {
                "Model": model_name,
                **calculate_metrics(y_test, predictions, probabilities),
            }
        )
        prediction_results[model_name] = {
            "predictions": predictions,
            "probabilities": probabilities,
        }

    comparison = pd.DataFrame(metrics_rows).set_index("Model")
    return comparison, prediction_results


def save_confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> None:
    """Save a labeled confusion matrix for the hybrid model."""
    matrix = confusion_matrix(actual, predicted)
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"],
        ax=axis,
    )
    axis.set_title("CNN-RNN-LSTM Confusion Matrix")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def save_roc_curve(
    actual: np.ndarray,
    comparison: pd.DataFrame,
    prediction_results: dict[str, dict[str, np.ndarray]],
) -> None:
    """Save ROC curves for the hybrid model and every baseline."""
    figure, axis = plt.subplots(figsize=(8, 6))
    for model_name, result in prediction_results.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            actual,
            result["probabilities"],
        )
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{model_name} (AUC={comparison.loc[model_name, 'ROC-AUC']:.3f})",
        )
    axis.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    axis.set_title("ROC Curve Comparison")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close(figure)


def save_precision_recall_curve(
    actual: np.ndarray,
    prediction_results: dict[str, dict[str, np.ndarray]],
) -> None:
    """Save precision-recall curves for all evaluated models."""
    figure, axis = plt.subplots(figsize=(8, 6))
    for model_name, result in prediction_results.items():
        precision, recall, _ = precision_recall_curve(
            actual,
            result["probabilities"],
        )
        axis.plot(recall, precision, linewidth=2, label=model_name)
    axis.set_title("Precision-Recall Curve Comparison")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(figure)


def save_comparison_table(comparison: pd.DataFrame) -> None:
    """Save the comparison as both CSV data and a screenshot-ready image."""
    comparison.round(4).to_csv(REPORTS_DIR / "model_comparison.csv")

    display_table = comparison.reset_index().copy()
    for column in display_table.columns[1:]:
        display_table[column] = display_table[column].map(lambda value: f"{value:.4f}")

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.axis("off")
    table = axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="center",
        loc="center",
        colColours=["#d9e8f5"] * len(display_table.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    axis.set_title("Model Performance Comparison", fontsize=14, pad=18)
    figure.tight_layout()
    figure.savefig(
        PLOTS_DIR / "model_comparison_table.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_text_reports(
    actual: np.ndarray,
    comparison: pd.DataFrame,
    prediction_results: dict[str, dict[str, np.ndarray]],
) -> None:
    """Save classification reports and a readable metric summary."""
    report_sections: list[str] = []
    for model_name, result in prediction_results.items():
        report_sections.extend(
            [
                model_name.upper(),
                "=" * len(model_name),
                classification_report(
                    actual,
                    result["predictions"],
                    target_names=["No Stroke", "Stroke"],
                    digits=4,
                    zero_division=0,
                ),
                "",
            ]
        )
    (REPORTS_DIR / "classification_report.txt").write_text(
        "\n".join(report_sections),
        encoding="utf-8",
    )
    (REPORTS_DIR / "evaluation_metrics.txt").write_text(
        comparison.round(4).to_string() + "\n",
        encoding="utf-8",
    )


def print_results(
    actual: np.ndarray,
    comparison: pd.DataFrame,
    prediction_results: dict[str, dict[str, np.ndarray]],
) -> None:
    """Print the hybrid classification report and complete comparison table."""
    print("\nCNN-RNN-LSTM CLASSIFICATION REPORT")
    print("=" * 80)
    print(
        classification_report(
            actual,
            prediction_results["CNN-RNN-LSTM"]["predictions"],
            target_names=["No Stroke", "Stroke"],
            digits=4,
            zero_division=0,
        )
    )
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    print(comparison.round(4).to_string())


def main() -> int:
    """Run the complete Phase 6 evaluation workflow."""
    configure_logging()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        x_train, y_train, x_test, y_test = load_evaluation_data()
        comparison, prediction_results = evaluate_all_models(
            x_train,
            y_train,
            x_test,
            y_test,
        )
        save_confusion_matrix(
            y_test,
            prediction_results["CNN-RNN-LSTM"]["predictions"],
        )
        save_roc_curve(y_test, comparison, prediction_results)
        save_precision_recall_curve(y_test, prediction_results)
        save_comparison_table(comparison)
        save_text_reports(y_test, comparison, prediction_results)
        print_results(y_test, comparison, prediction_results)
    except (FileNotFoundError, PermissionError, ValueError, OSError) as error:
        LOGGER.error("Model evaluation failed: %s", error)
        return 1

    LOGGER.info("Phase 6 model evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
