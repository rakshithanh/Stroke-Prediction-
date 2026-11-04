"""Generate dissertation visualizations for exploratory data analysis.

The plots use the original stroke dataset. No model training or prediction is
performed in this module.
"""

import logging
import os
import textwrap
from pathlib import Path

# Avoid a harmless joblib CPU-detection warning on this Windows environment.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.data_loader import (
    DEFAULT_DATASET_PATH,
    PROJECT_ROOT,
    configure_logging,
    load_stroke_dataset,
)


LOGGER = logging.getLogger("exploratory_data_analysis")
PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"
INTERPRETATIONS_PATH = PROJECT_ROOT / "reports" / "eda_interpretations.txt"
COLORS = ["#1f5a91", "#dc3545"]
RANDOM_STATE = 42


def percentage(value: float) -> str:
    """Format a decimal proportion as a percentage."""
    return f"{value * 100:.2f}%"


def save_plot(
    figure: plt.Figure,
    filename: str,
    interpretation: str,
) -> None:
    """Add interpretation below a plot and save the figure and its text."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    wrapped_text = textwrap.fill(
        f"Interpretation: {interpretation}",
        width=130,
    )
    figure.text(
        0.5,
        0.025,
        wrapped_text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f2f4f7"},
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(figure)

    (PLOTS_DIR / filename.replace(".png", "_interpretation.txt")).write_text(
        interpretation,
        encoding="utf-8",
    )
    LOGGER.info("Saved plot: %s", filename)


def plot_stroke_distribution(dataframe: pd.DataFrame) -> str:
    """Plot the target class counts."""
    counts = dataframe["stroke"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(8, 6))
    bars = axis.bar(["No Stroke (0)", "Stroke (1)"], counts, color=COLORS)
    axis.set_title("Stroke Distribution")
    axis.set_xlabel("Stroke Class")
    axis.set_ylabel("Number of Records")
    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({count / len(dataframe) * 100:.2f}%)",
            ha="center",
            va="bottom",
        )

    interpretation = (
        f"Only {counts.get(1, 0)} of {len(dataframe)} records "
        f"({counts.get(1, 0) / len(dataframe) * 100:.2f}%) represent stroke. "
        "The original dataset is therefore strongly class-imbalanced."
    )
    save_plot(figure, "01_stroke_distribution.png", interpretation)
    return interpretation


def plot_numeric_distribution(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    label: str,
) -> str:
    """Plot a numerical feature by stroke class."""
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.histplot(
        data=dataframe,
        x=column,
        hue="stroke",
        bins=30,
        kde=True,
        stat="density",
        common_norm=False,
        palette=COLORS,
        alpha=0.45,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(label)
    axis.set_ylabel("Density")

    medians = dataframe.groupby("stroke")[column].median()
    interpretation = (
        f"The median {label.lower()} is {medians.get(0, float('nan')):.2f} "
        f"for non-stroke records and {medians.get(1, float('nan')):.2f} "
        "for stroke records. The curves describe association and do not by "
        "themselves establish causation."
    )
    save_plot(figure, filename, interpretation)
    return interpretation


def plot_correlation_heatmap(dataframe: pd.DataFrame) -> str:
    """Plot correlations among numerical variables, excluding patient ID."""
    numeric_data = dataframe.select_dtypes(include="number").drop(
        columns=["id"],
        errors="ignore",
    )
    correlations = numeric_data.corr()
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title("Correlation Heatmap")

    stroke_correlations = correlations["stroke"].drop("stroke")
    strongest_feature = stroke_correlations.abs().idxmax()
    strongest_value = stroke_correlations[strongest_feature]
    interpretation = (
        f"Among the numerical predictors, '{strongest_feature}' has the "
        f"largest absolute linear correlation with stroke ({strongest_value:.2f}). "
        "Correlation values show linear association, not causation."
    )
    save_plot(figure, "05_correlation_heatmap.png", interpretation)
    return interpretation


def plot_category_stroke_rate(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    label: str,
) -> str:
    """Plot the observed stroke percentage for each category."""
    summary = (
        dataframe.groupby(column, dropna=False)["stroke"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.bar(
        summary.index.astype(str),
        summary["mean"] * 100,
        color="#1f77b4",
    )
    axis.set_title(title)
    axis.set_xlabel(label)
    axis.set_ylabel("Observed Stroke Rate (%)")
    axis.tick_params(axis="x", rotation=20)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['mean'] * 100:.2f}%\n(n={int(row['count'])})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    top_category = str(summary.index[0])
    top_rate = summary.iloc[0]["mean"]
    top_count = int(summary.iloc[0]["count"])
    interpretation = (
        f"'{top_category}' has the highest observed stroke percentage "
        f"({percentage(top_rate)}) in this dataset, based on {top_count} records. "
        "Category sizes differ, so the percentages should be interpreted cautiously."
    )
    save_plot(figure, filename, interpretation)
    return interpretation


def plot_pair_plot(dataframe: pd.DataFrame) -> str:
    """Create a feasible pair plot using a reproducible stratified sample."""
    stroke_rows = dataframe[dataframe["stroke"] == 1]
    non_stroke_rows = dataframe[dataframe["stroke"] == 0].sample(
        n=min(1000, (dataframe["stroke"] == 0).sum()),
        random_state=RANDOM_STATE,
    )
    pair_data = pd.concat([stroke_rows, non_stroke_rows], ignore_index=True)
    pair_data = pair_data[
        ["age", "avg_glucose_level", "bmi", "stroke"]
    ].dropna()
    pair_data["stroke"] = pair_data["stroke"].map(
        {0: "No Stroke", 1: "Stroke"}
    )

    grid = sns.pairplot(
        pair_data,
        vars=["age", "avg_glucose_level", "bmi"],
        hue="stroke",
        palette={"No Stroke": COLORS[0], "Stroke": COLORS[1]},
        diag_kind="hist",
        plot_kws={"alpha": 0.45, "s": 22},
    )
    grid.figure.suptitle(
        "Pair Plot of Selected Numerical Features",
        y=1.02,
    )
    interpretation = (
        f"The pair plot uses {len(pair_data)} complete records: all available "
        "stroke records plus a reproducible sample of non-stroke records. "
        "The classes overlap across age, glucose, and BMI, although stroke "
        "observations are more concentrated at older ages."
    )
    save_plot(grid.figure, "10_pair_plot.png", interpretation)
    return interpretation


def generate_all_plots(dataframe: pd.DataFrame) -> dict[str, str]:
    """Generate every visualization requested for Phase 4."""
    sns.set_theme(style="whitegrid", context="notebook")
    interpretations = {
        "Stroke Distribution": plot_stroke_distribution(dataframe),
        "Age Distribution": plot_numeric_distribution(
            dataframe,
            "age",
            "Age Distribution by Stroke Class",
            "02_age_distribution.png",
            "Age",
        ),
        "BMI Distribution": plot_numeric_distribution(
            dataframe,
            "bmi",
            "BMI Distribution by Stroke Class",
            "03_bmi_distribution.png",
            "BMI",
        ),
        "Glucose Distribution": plot_numeric_distribution(
            dataframe,
            "avg_glucose_level",
            "Average Glucose Distribution by Stroke Class",
            "04_glucose_distribution.png",
            "Average Glucose Level",
        ),
        "Correlation Heatmap": plot_correlation_heatmap(dataframe),
        "Gender Analysis": plot_category_stroke_rate(
            dataframe,
            "gender",
            "Stroke Rate by Gender",
            "06_gender_analysis.png",
            "Gender",
        ),
        "Smoking Analysis": plot_category_stroke_rate(
            dataframe,
            "smoking_status",
            "Stroke Rate by Smoking Status",
            "07_smoking_analysis.png",
            "Smoking Status",
        ),
        "Hypertension Analysis": plot_category_stroke_rate(
            dataframe.assign(
                hypertension=dataframe["hypertension"].map({0: "No", 1: "Yes"})
            ),
            "hypertension",
            "Stroke Rate by Hypertension Status",
            "08_hypertension_analysis.png",
            "Hypertension",
        ),
        "Heart Disease Analysis": plot_category_stroke_rate(
            dataframe.assign(
                heart_disease=dataframe["heart_disease"].map(
                    {0: "No", 1: "Yes"}
                )
            ),
            "heart_disease",
            "Stroke Rate by Heart Disease Status",
            "09_heart_disease_analysis.png",
            "Heart Disease",
        ),
        "Pair Plot": plot_pair_plot(dataframe),
    }
    return interpretations


def save_interpretation_report(interpretations: dict[str, str]) -> None:
    """Save all plot interpretations in one dissertation-friendly report."""
    sections = ["EXPLORATORY DATA ANALYSIS INTERPRETATIONS", "=" * 46]
    for index, (title, interpretation) in enumerate(
        interpretations.items(),
        start=1,
    ):
        sections.extend(
            [
                "",
                f"{index}. {title}",
                "-" * (len(title) + 3),
                interpretation,
            ]
        )
    INTERPRETATIONS_PATH.write_text("\n".join(sections), encoding="utf-8")
    LOGGER.info("Saved interpretation report: %s", INTERPRETATIONS_PATH)


def main() -> int:
    """Load the dataset and generate all Phase 4 outputs."""
    configure_logging()
    try:
        dataframe = load_stroke_dataset(DEFAULT_DATASET_PATH)
        interpretations = generate_all_plots(dataframe)
        save_interpretation_report(interpretations)
    except (FileNotFoundError, PermissionError, ValueError, OSError) as error:
        LOGGER.error("EDA generation failed: %s", error)
        return 1

    LOGGER.info("Phase 4 exploratory data analysis completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
