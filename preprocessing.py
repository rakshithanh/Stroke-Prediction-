"""Preprocess the stroke dataset for the academic project.

This phase prepares data only. It deliberately contains no model creation,
training, evaluation, or prediction code.
"""

import logging
import os
from pathlib import Path

# This Windows environment does not report physical cores to joblib. Setting a
# local fallback prevents a harmless warning during the academic demonstration.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib

# Use a non-interactive backend so the graph can be saved without opening a UI.
matplotlib.use("Agg")

import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.data_loader import (
    DEFAULT_DATASET_PATH,
    PROJECT_ROOT,
    TARGET_COLUMN,
    configure_logging,
    load_stroke_dataset,
)

LOGGER = logging.getLogger("data_preprocessing")
PROCESSED_DATASET_PATH = PROJECT_ROOT / "dataset" / "processed_dataset.csv"
PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.20
IDENTIFIER_COLUMNS = ["id"]


def remove_duplicate_records(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exact duplicate records removed."""
    duplicate_count = int(dataframe.duplicated().sum())
    LOGGER.info("Duplicate records found: %d", duplicate_count)
    return dataframe.drop_duplicates().reset_index(drop=True)


def select_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Separate predictors and target, excluding the patient identifier."""
    excluded_columns = [
        column for column in IDENTIFIER_COLUMNS if column in dataframe.columns
    ]
    features = dataframe.drop(columns=excluded_columns + [TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN]

    numerical_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    return features, target, numerical_columns, categorical_columns


def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Create imputing, encoding, and normalization transformations."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        verbose_feature_names_out=False,
    )


def prepare_data(dataframe: pd.DataFrame) -> dict[str, object]:
    """Split, transform, and balance the dataset without data leakage."""
    cleaned_data = remove_duplicate_records(dataframe)
    features, target, numerical_columns, categorical_columns = select_features(
        cleaned_data
    )

    # Split first so preprocessing learns values from training data only.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns,
    )
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    # SMOTE is applied only to the training set. The test set remains realistic.
    smote = SMOTE(random_state=RANDOM_STATE)
    x_train_balanced, y_train_balanced = smote.fit_resample(
        x_train_transformed,
        y_train,
    )

    return {
        "cleaned_data": cleaned_data,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "x_train_balanced": x_train_balanced,
        "x_test": x_test_transformed,
        "y_train_original": y_train,
        "y_train_balanced": y_train_balanced,
        "y_test": y_test.reset_index(drop=True),
    }


def create_processed_dataset(result: dict[str, object]) -> pd.DataFrame:
    """Combine processed train and test rows with a clear split label."""
    train_data = pd.DataFrame(
        result["x_train_balanced"],
        columns=result["feature_names"],
    )
    train_data[TARGET_COLUMN] = result["y_train_balanced"].to_numpy()
    train_data["data_split"] = "train"

    test_data = pd.DataFrame(
        result["x_test"],
        columns=result["feature_names"],
    )
    test_data[TARGET_COLUMN] = result["y_test"].to_numpy()
    test_data["data_split"] = "test"

    return pd.concat([train_data, test_data], ignore_index=True)


def save_class_balance_graph(result: dict[str, object]) -> None:
    """Save the before-and-after SMOTE class distribution graph."""
    original_counts = (
        result["y_train_original"].value_counts().sort_index().reindex([0, 1])
    )
    balanced_counts = (
        result["y_train_balanced"].value_counts().sort_index().reindex([0, 1])
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#174a7e", "#dc3545"]

    axes[0].bar(["No Stroke (0)", "Stroke (1)"], original_counts, color=colors)
    axes[0].set_title("Training Classes Before SMOTE")
    axes[0].set_ylabel("Number of Records")

    axes[1].bar(["No Stroke (0)", "Stroke (1)"], balanced_counts, color=colors)
    axes[1].set_title("Training Classes After SMOTE")

    for axis in axes:
        for bar in axis.patches:
            axis.annotate(
                str(int(bar.get_height())),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
            )

    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "class_balance_graph.png", dpi=150)
    plt.close(figure)


def save_outputs(
    original_data: pd.DataFrame,
    result: dict[str, object],
    processed_data: pd.DataFrame,
) -> None:
    """Save the processed CSV, fitted pipeline, and Phase 3 reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed_data.to_csv(PROCESSED_DATASET_PATH, index=False)
    joblib.dump(
        {
            "preprocessor": result["preprocessor"],
            "feature_names": result["feature_names"],
            "target_column": TARGET_COLUMN,
            "excluded_columns": IDENTIFIER_COLUMNS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        PIPELINE_PATH,
    )

    before_summary = [
        "BEFORE PREPROCESSING",
        "=" * 40,
        f"Shape: {original_data.shape}",
        f"Duplicate records: {original_data.duplicated().sum()}",
        "",
        "Missing values:",
        original_data.isna().sum().to_string(),
        "",
        "Class distribution:",
        original_data[TARGET_COLUMN].value_counts().sort_index().to_string(),
    ]
    (REPORTS_DIR / "before_preprocessing.txt").write_text(
        "\n".join(before_summary),
        encoding="utf-8",
    )

    after_summary = [
        "AFTER PREPROCESSING",
        "=" * 40,
        f"Processed shape: {processed_data.shape}",
        f"Training rows after SMOTE: {len(result['y_train_balanced'])}",
        f"Untouched test rows: {len(result['y_test'])}",
        f"Target column: {TARGET_COLUMN}",
        f"Features used ({len(result['feature_names'])}):",
        ", ".join(result["feature_names"]),
        "",
        "Missing values in processed dataset:",
        processed_data.isna().sum().to_string(),
    ]
    (REPORTS_DIR / "after_preprocessing.txt").write_text(
        "\n".join(after_summary),
        encoding="utf-8",
    )

    smote_summary = [
        "SMOTE RESULT (TRAINING SET ONLY)",
        "=" * 40,
        "Before SMOTE:",
        result["y_train_original"].value_counts().sort_index().to_string(),
        "",
        "After SMOTE:",
        result["y_train_balanced"].value_counts().sort_index().to_string(),
    ]
    (REPORTS_DIR / "smote_result.txt").write_text(
        "\n".join(smote_summary),
        encoding="utf-8",
    )
    save_class_balance_graph(result)
    LOGGER.info("Processed dataset saved: %s", PROCESSED_DATASET_PATH)
    LOGGER.info("Preprocessing pipeline saved: %s", PIPELINE_PATH)


def print_results(
    original_data: pd.DataFrame,
    result: dict[str, object],
    processed_data: pd.DataFrame,
) -> None:
    """Print the requested Phase 3 values for screenshots."""
    print("\nBEFORE PREPROCESSING")
    print("=" * 70)
    print(f"Original Shape: {original_data.shape}")
    print(original_data.head(10).to_string(index=False))

    print("\nAFTER PREPROCESSING")
    print("=" * 70)
    print(f"Processed Shape: {processed_data.shape}")
    print(f"Features Used ({len(result['feature_names'])}):")
    print(", ".join(result["feature_names"]))
    print(f"Target Column: {TARGET_COLUMN}")
    print(f"Missing Values After Processing: {processed_data.isna().sum().sum()}")

    print("\nSMOTE RESULT (TRAINING SET ONLY)")
    print("=" * 70)
    print("Before SMOTE:")
    print(result["y_train_original"].value_counts().sort_index().to_string())
    print("\nAfter SMOTE:")
    print(result["y_train_balanced"].value_counts().sort_index().to_string())


def main() -> int:
    """Run Phase 3 preprocessing and save all requested outputs."""
    configure_logging()
    try:
        original_data = load_stroke_dataset(DEFAULT_DATASET_PATH)
        result = prepare_data(original_data)
        processed_data = create_processed_dataset(result)
        save_outputs(original_data, result, processed_data)
        print_results(original_data, result, processed_data)
    except (FileNotFoundError, PermissionError, ValueError, OSError) as error:
        LOGGER.error("Preprocessing failed: %s", error)
        return 1

    LOGGER.info("Phase 3 preprocessing completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
