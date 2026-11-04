"""Generate SHAP explanations for the trained CNN-RNN-LSTM model."""

import logging
import os
from pathlib import Path

# Keep TensorFlow and parallel-library output concise for screenshots.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tensorflow import keras


LOGGER = logging.getLogger("shap_explanations")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_hybrid_model.keras"
PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
DATASET_PATH = PROJECT_ROOT / "dataset" / "processed_dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
TOP_FEATURES_PATH = REPORTS_DIR / "shap_top_features.txt"
TOP_FEATURES_CSV_PATH = REPORTS_DIR / "shap_top_features.csv"
SHAP_VALUES_PATH = REPORTS_DIR / "shap_values.npz"
TARGET_COLUMN = "stroke"
SPLIT_COLUMN = "data_split"
BACKGROUND_SIZE = 30
EXPLANATION_SIZE = 50
SHAP_SAMPLES = 100
RANDOM_STATE = 42


def configure_logging() -> None:
    """Configure clean, readable SHAP generation logs."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # Kernel SHAP emits detailed solver internals at INFO level. They are useful
    # for library debugging but not for an academic demonstration screenshot.
    logging.getLogger("shap").setLevel(logging.WARNING)


def load_explanation_inputs() -> tuple[
    keras.Model,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Load the trained model, background data, test records, and features."""
    for required_path in (MODEL_PATH, PIPELINE_PATH, DATASET_PATH):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required Phase 8 input missing: {required_path}")

    # Compilation is unnecessary because SHAP only requires inference.
    model = keras.models.load_model(MODEL_PATH, compile=False)
    preprocessing_artifact = joblib.load(PIPELINE_PATH)
    feature_names = preprocessing_artifact["feature_names"]

    dataframe = pd.read_csv(DATASET_PATH)
    training_data = dataframe[dataframe[SPLIT_COLUMN] == "train"]
    test_data = dataframe[dataframe[SPLIT_COLUMN] == "test"]
    if training_data.empty or test_data.empty:
        raise ValueError("Processed training and test records are required.")

    background_data = training_data[feature_names].sample(
        n=min(BACKGROUND_SIZE, len(training_data)),
        random_state=RANDOM_STATE,
    )
    explanation_data = test_data[feature_names].sample(
        n=min(EXPLANATION_SIZE, len(test_data)),
        random_state=RANDOM_STATE,
    )
    LOGGER.info("Loaded trained model: %s", MODEL_PATH.name)
    LOGGER.info("SHAP background records: %d", len(background_data))
    LOGGER.info("Records to explain: %d", len(explanation_data))
    LOGGER.info("Features: %d", len(feature_names))
    return (
        model,
        background_data.to_numpy(dtype=np.float32),
        explanation_data.to_numpy(dtype=np.float32),
        feature_names,
    )


def create_prediction_function(model: keras.Model):
    """Wrap 2D tabular values for the model's required 3D sequence input."""

    def predict_stroke_probability(features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        sequence_features = features.reshape(
            features.shape[0],
            features.shape[1],
            1,
        )
        probabilities = model(sequence_features, training=False).numpy()
        return probabilities.reshape(-1)

    return predict_stroke_probability


def calculate_shap_values(
    model: keras.Model,
    background_data: np.ndarray,
    explanation_data: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Calculate model-agnostic SHAP values for selected test records."""
    LOGGER.info("Calculating SHAP values; this may take a few minutes.")
    prediction_function = create_prediction_function(model)
    explainer = shap.KernelExplainer(prediction_function, background_data)
    raw_values = explainer.shap_values(
        explanation_data,
        nsamples=SHAP_SAMPLES,
        silent=True,
    )

    # Normalize SHAP's possible binary-output formats to records x features.
    if isinstance(raw_values, list):
        raw_values = raw_values[0]
    shap_values = np.asarray(raw_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]
    if shap_values.shape != explanation_data.shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; "
            f"expected {explanation_data.shape}."
        )

    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    LOGGER.info("SHAP values calculated successfully.")
    return shap_values, expected_value


def save_current_figure(filename: str) -> None:
    """Save and close the active Matplotlib SHAP figure."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure = plt.gcf()
    figure.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved SHAP plot: %s", filename)


def generate_summary_plot(
    shap_values: np.ndarray,
    explanation_data: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save the SHAP beeswarm summary plot."""
    plt.figure()
    shap.summary_plot(
        shap_values,
        explanation_data,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 7),
    )
    plt.title("SHAP Summary Plot")
    save_current_figure("shap_summary_plot.png")


def generate_bar_plot(
    shap_values: np.ndarray,
    explanation_data: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save global mean absolute feature importance as a SHAP bar plot."""
    plt.figure()
    shap.summary_plot(
        shap_values,
        explanation_data,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=(10, 7),
    )
    plt.title("SHAP Global Feature Importance")
    save_current_figure("shap_bar_plot.png")


def generate_force_plot(
    shap_values: np.ndarray,
    expected_value: float,
    explanation_data: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save a force plot explaining the first selected test record."""
    shap.force_plot(
        expected_value,
        shap_values[0],
        explanation_data[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        text_rotation=15,
    )
    plt.title("SHAP Force Plot - One Test Record", pad=20)
    save_current_figure("shap_force_plot.png")


def generate_waterfall_plot(
    shap_values: np.ndarray,
    expected_value: float,
    explanation_data: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save a waterfall explanation for the first selected test record."""
    single_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=explanation_data[0],
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(single_explanation, max_display=12, show=False)
    plt.title("SHAP Waterfall Plot - One Test Record")
    save_current_figure("shap_waterfall_plot.png")


def generate_dependence_plot(
    shap_values: np.ndarray,
    explanation_data: np.ndarray,
    feature_names: list[str],
    top_feature: str,
) -> None:
    """Save dependence between the top feature value and its SHAP impact."""
    plt.figure(figsize=(9, 6))
    shap.dependence_plot(
        top_feature,
        shap_values,
        explanation_data,
        feature_names=feature_names,
        interaction_index=None,
        show=False,
    )
    plt.title(f"SHAP Dependence Plot - {top_feature}")
    save_current_figure("shap_dependence_plot.png")


def save_top_feature_report(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> str:
    """Rank features and explain magnitude and average contribution direction."""
    mean_absolute_values = np.abs(shap_values).mean(axis=0)
    mean_signed_values = shap_values.mean(axis=0)
    ranking = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean Absolute SHAP": mean_absolute_values,
            "Mean Signed SHAP": mean_signed_values,
        }
    ).sort_values("Mean Absolute SHAP", ascending=False, ignore_index=True)
    ranking.to_csv(TOP_FEATURES_CSV_PATH, index=False)

    report_lines = [
        "TOP CONTRIBUTING FEATURES",
        "=" * 40,
        "Mean absolute SHAP measures contribution strength. Positive signed SHAP",
        "raises the model's stroke probability; negative signed SHAP lowers it.",
        "Values describe this model and sample, not medical causation.",
        "",
    ]
    for rank, row in ranking.head(10).iterrows():
        if row["Mean Signed SHAP"] > 0:
            direction = "increased predicted stroke probability on average"
        elif row["Mean Signed SHAP"] < 0:
            direction = "decreased predicted stroke probability on average"
        else:
            direction = "had no average directional effect"
        report_lines.append(
            f"{rank + 1}. {row['Feature']}: mean |SHAP|="
            f"{row['Mean Absolute SHAP']:.6f}; {direction}."
        )
    TOP_FEATURES_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report_lines))
    return str(ranking.iloc[0]["Feature"])


def main() -> int:
    """Generate all requested Phase 8 SHAP explanations."""
    configure_logging()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model, background_data, explanation_data, feature_names = (
            load_explanation_inputs()
        )
        shap_values, expected_value = calculate_shap_values(
            model,
            background_data,
            explanation_data,
        )
        top_feature = save_top_feature_report(shap_values, feature_names)
        generate_summary_plot(shap_values, explanation_data, feature_names)
        generate_bar_plot(shap_values, explanation_data, feature_names)
        generate_force_plot(
            shap_values,
            expected_value,
            explanation_data,
            feature_names,
        )
        generate_waterfall_plot(
            shap_values,
            expected_value,
            explanation_data,
            feature_names,
        )
        generate_dependence_plot(
            shap_values,
            explanation_data,
            feature_names,
            top_feature,
        )
        np.savez_compressed(
            SHAP_VALUES_PATH,
            shap_values=shap_values,
            explanation_data=explanation_data,
            expected_value=expected_value,
            feature_names=np.asarray(feature_names),
        )
    except (FileNotFoundError, PermissionError, ValueError, OSError) as error:
        LOGGER.error("SHAP explanation generation failed: %s", error)
        return 1

    LOGGER.info("Phase 8 SHAP explanation generation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
