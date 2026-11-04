"""Load and validate the stroke dataset for the academic project.

This module performs descriptive validation only. It does not preprocess data,
train a model, or make predictions.
"""

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


LOGGER = logging.getLogger("dataset_validation")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "healthcare-dataset-stroke-data.csv"
)
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_COLUMN = "stroke"


def configure_logging() -> None:
    """Configure short, readable console logs for the academic demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def load_stroke_dataset(dataset_path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the stroke CSV and raise a clear error for common problems."""
    dataset_path = Path(dataset_path)
    LOGGER.info("Loading dataset: %s", dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Place healthcare-dataset-stroke-data.csv in the dataset folder."
        )

    try:
        dataframe = pd.read_csv(dataset_path)
    except EmptyDataError as error:
        raise ValueError("The dataset file is empty.") from error
    except ParserError as error:
        raise ValueError("The dataset CSV format is invalid.") from error
    except UnicodeDecodeError as error:
        raise ValueError("The dataset text encoding could not be read.") from error
    except PermissionError as error:
        raise PermissionError(
            "Permission denied while reading the dataset. Close any program "
            "locking the file and try again."
        ) from error

    if dataframe.empty:
        raise ValueError("The dataset contains no records.")
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing from the dataset."
        )

    LOGGER.info("Dataset loaded successfully.")
    return dataframe


def create_dataset_report(dataframe: pd.DataFrame) -> dict[str, object]:
    """Calculate the validation outputs requested in Phase 2."""
    info_buffer = StringIO()
    dataframe.info(buf=info_buffer)

    class_counts = dataframe[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    class_distribution = pd.DataFrame(
        {
            "count": class_counts,
            "percentage": (class_counts / len(dataframe) * 100).round(2),
        }
    )
    class_distribution.index.name = TARGET_COLUMN

    return {
        "row_count": dataframe.shape[0],
        "column_count": dataframe.shape[1],
        "column_names": dataframe.columns.tolist(),
        "data_types": dataframe.dtypes.astype(str),
        "missing_values": dataframe.isna().sum(),
        "duplicate_records": int(dataframe.duplicated().sum()),
        "class_distribution": class_distribution,
        "basic_statistics": dataframe.describe(include="all").transpose(),
        "data_information": info_buffer.getvalue(),
        "preview": dataframe.head(10),
    }


def save_dataset_reports(
    dataframe: pd.DataFrame,
    report: dict[str, object],
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> None:
    """Save readable text and CSV reports in the reports folder."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report["preview"].to_csv(reports_dir / "dataset_preview.csv", index=False)
    report["data_types"].rename("data_type").to_csv(
        reports_dir / "dataset_data_types.csv"
    )
    report["missing_values"].rename("missing_count").to_csv(
        reports_dir / "missing_values.csv"
    )
    report["class_distribution"].to_csv(
        reports_dir / "class_distribution.csv"
    )
    report["basic_statistics"].to_csv(
        reports_dir / "basic_statistics.csv"
    )
    (reports_dir / "dataset_info.txt").write_text(
        report["data_information"],
        encoding="utf-8",
    )

    summary_lines = [
        "STROKE DATASET VALIDATION SUMMARY",
        "=" * 34,
        f"Rows: {report['row_count']}",
        f"Columns: {report['column_count']}",
        f"Column names: {', '.join(report['column_names'])}",
        f"Duplicate records: {report['duplicate_records']}",
        "",
        "DATA TYPES",
        str(report["data_types"]),
        "",
        "MISSING VALUES",
        str(report["missing_values"]),
        "",
        "CLASS DISTRIBUTION",
        str(report["class_distribution"]),
        "",
        "DATASET PREVIEW",
        dataframe.head(10).to_string(index=False),
    ]
    (reports_dir / "dataset_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )
    LOGGER.info("Reports saved to: %s", reports_dir)


def print_dataset_report(dataframe: pd.DataFrame, report: dict[str, object]) -> None:
    """Print clean Phase 2 results for terminal screenshots."""
    print("\nDATASET PREVIEW")
    print("=" * 80)
    print(dataframe.head(10).to_string(index=False))
    print(f"\nRows: {report['row_count']}")
    print(f"Columns: {report['column_count']}")
    print(f"Column names: {', '.join(report['column_names'])}")

    print("\nDATA TYPES")
    print("=" * 80)
    print(report["data_types"].to_string())

    print("\nDATASET INFO")
    print("=" * 80)
    print(report["data_information"].rstrip())

    print("\nMISSING VALUES")
    print("=" * 80)
    print(report["missing_values"].to_string())
    print(f"\nDuplicate records: {report['duplicate_records']}")

    print("\nCLASS DISTRIBUTION")
    print("=" * 80)
    print(report["class_distribution"].to_string())

    print("\nBASIC STATISTICS")
    print("=" * 80)
    print(report["basic_statistics"].to_string())


def main() -> int:
    """Run dataset loading, validation, display, and report generation."""
    configure_logging()

    try:
        dataframe = load_stroke_dataset()
        report = create_dataset_report(dataframe)
        print_dataset_report(dataframe, report)
        save_dataset_reports(dataframe, report)
    except (FileNotFoundError, PermissionError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    except OSError as error:
        LOGGER.error("A file-system error occurred: %s", error)
        return 1

    LOGGER.info("Phase 2 dataset validation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
