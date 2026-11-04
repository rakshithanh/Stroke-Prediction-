"""Convert a model probability into an academic stroke-risk category.

This module presents model output in a simple form. It is not a medical
diagnosis and must not be used as a substitute for professional advice.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOGGER = logging.getLogger("dynamic_risk_assessment")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
RISK_REPORT_PATH = REPORTS_DIR / "risk_categories.txt"
LOW_RISK_LIMIT = 0.30
HIGH_RISK_LIMIT = 0.70


@dataclass(frozen=True)
class RiskAssessment:
    """Store display-ready output from the dynamic risk assessment."""

    probability: float
    risk_percentage: float
    category: str
    color_name: str
    color_hex: str
    bootstrap_class: str
    message: str
    disclaimer: str = "Academic prediction only; this is not a medical diagnosis."

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary that can later be passed to a Flask template."""
        return asdict(self)


def validate_probability(probability: float) -> float:
    """Validate that model probability is numeric and between zero and one."""
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise TypeError("Prediction probability must be a numeric value.")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Prediction probability must be between 0 and 1.")
    return probability


def assess_risk(probability: float) -> RiskAssessment:
    """Classify prediction probability as Low, Medium, or High Risk.

    Thresholds used for this academic demonstration:
    - Low Risk: probability below 30%
    - Medium Risk: probability from 30% up to, but not including, 70%
    - High Risk: probability of 70% or above
    """
    probability = validate_probability(probability)

    if probability < LOW_RISK_LIMIT:
        category = "Low Risk"
        color_name = "Green"
        color_hex = "#198754"
        bootstrap_class = "success"
        message = (
            "The model estimates a lower stroke risk. Maintain healthy habits "
            "and continue regular health check-ups."
        )
    elif probability < HIGH_RISK_LIMIT:
        category = "Medium Risk"
        color_name = "Amber"
        color_hex = "#ffc107"
        bootstrap_class = "warning"
        message = (
            "The model estimates a moderate stroke risk. Review relevant "
            "lifestyle factors and consider professional medical guidance."
        )
    else:
        category = "High Risk"
        color_name = "Red"
        color_hex = "#dc3545"
        bootstrap_class = "danger"
        message = (
            "The model estimates a higher stroke risk. Seek timely assessment "
            "from a qualified healthcare professional."
        )

    return RiskAssessment(
        probability=probability,
        risk_percentage=round(probability * 100, 2),
        category=category,
        color_name=color_name,
        color_hex=color_hex,
        bootstrap_class=bootstrap_class,
        message=message,
    )


def save_risk_card(assessment: RiskAssessment, filename: str) -> None:
    """Save one screenshot-ready card for a risk assessment example."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    text_color = "#212529" if assessment.category == "Medium Risk" else "white"

    figure, axis = plt.subplots(figsize=(9, 4.5))
    figure.patch.set_facecolor("white")
    axis.axis("off")
    axis.text(
        0.5,
        0.82,
        assessment.category,
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
        color=text_color,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": assessment.color_hex,
            "edgecolor": assessment.color_hex,
        },
    )
    axis.text(
        0.5,
        0.55,
        f"Risk: {assessment.risk_percentage:.2f}%",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=assessment.color_hex,
    )
    axis.text(
        0.5,
        0.34,
        assessment.message,
        ha="center",
        va="center",
        fontsize=13,
        wrap=True,
    )
    axis.text(
        0.5,
        0.12,
        assessment.disclaimer,
        ha="center",
        va="center",
        fontsize=10,
        color="#6c757d",
        style="italic",
    )
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved risk screenshot: %s", filename)


def generate_risk_examples() -> list[RiskAssessment]:
    """Generate one representative output for each risk category."""
    example_probabilities = [0.15, 0.50, 0.85]
    filenames = ["risk_low.png", "risk_medium.png", "risk_high.png"]
    assessments = [assess_risk(value) for value in example_probabilities]

    for assessment, filename in zip(assessments, filenames):
        save_risk_card(assessment, filename)

    report_lines = [
        "DYNAMIC RISK ASSESSMENT CATEGORIES",
        "=" * 42,
        "Low Risk: probability < 30%",
        "Medium Risk: probability >= 30% and < 70%",
        "High Risk: probability >= 70%",
        "",
    ]
    for assessment in assessments:
        report_lines.extend(
            [
                assessment.category.upper(),
                f"Risk Percentage: {assessment.risk_percentage:.2f}%",
                f"Risk Color: {assessment.color_name} ({assessment.color_hex})",
                f"Risk Message: {assessment.message}",
                f"Disclaimer: {assessment.disclaimer}",
                "",
            ]
        )
    RISK_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    return assessments


def main() -> int:
    """Print and save the three Phase 9 risk-category examples."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        assessments = generate_risk_examples()
    except (TypeError, ValueError, PermissionError, OSError) as error:
        LOGGER.error("Risk assessment generation failed: %s", error)
        return 1

    print("\nDYNAMIC RISK ASSESSMENT")
    print("=" * 70)
    for assessment in assessments:
        print(f"\nCategory: {assessment.category}")
        print(f"Risk %: {assessment.risk_percentage:.2f}%")
        print(f"Risk Color: {assessment.color_name} ({assessment.color_hex})")
        print(f"Risk Message: {assessment.message}")
    LOGGER.info("Phase 9 dynamic risk assessment completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
