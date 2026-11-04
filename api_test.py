"""Repeatable Phase 11 API demonstration using Flask's local test client."""

import json
from pathlib import Path

from app import create_app


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "api_test_results.txt"
DEMO_EMAIL = "phase11.demo@example.com"
DEMO_PASSWORD = "academic123"


def pretty(data: dict) -> str:
    """Format API JSON for terminal and report screenshots."""
    return json.dumps(data, indent=2, sort_keys=True)


def main() -> int:
    """Call every required API and verify its HTTP response."""
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()
    output_lines = ["PHASE 11 FLASK API TESTING", "=" * 70]

    register_response = client.post(
        "/api/register",
        json={
            "name": "Academic Demo User",
            "email": DEMO_EMAIL,
            "password": DEMO_PASSWORD,
        },
    )
    if register_response.status_code not in {201, 409}:
        raise RuntimeError("Register API failed.")
    output_lines.extend(
        [
            "\nREGISTER API",
            f"Status: {register_response.status_code}",
            pretty(register_response.get_json()),
        ]
    )

    login_response = client.post(
        "/api/login",
        json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD},
    )
    if login_response.status_code != 200:
        raise RuntimeError("Login API failed.")
    output_lines.extend(
        [
            "\nLOGIN API",
            f"Status: {login_response.status_code}",
            pretty(login_response.get_json()),
        ]
    )

    patient_input = {
        "gender": "Male",
        "age": 67,
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked",
    }
    prediction_response = client.post("/api/predict", json=patient_input)
    if prediction_response.status_code != 200:
        raise RuntimeError(
            f"Prediction API failed: {prediction_response.get_data(as_text=True)}"
        )
    prediction_json = prediction_response.get_json()
    prediction_summary = {
        "success": prediction_json["success"],
        "message": prediction_json["message"],
        "history_id": prediction_json["history_id"],
        "prediction_probability": prediction_json["prediction_probability"],
        "risk": prediction_json["risk"],
        "recommendation_sections": list(
            prediction_json["recommendations"].keys()
        ),
    }
    output_lines.extend(
        [
            "\nPREDICTION API",
            f"Status: {prediction_response.status_code}",
            pretty(prediction_summary),
        ]
    )

    history_response = client.get("/api/history")
    if history_response.status_code != 200:
        raise RuntimeError("History API failed.")
    history_json = history_response.get_json()
    output_lines.extend(
        [
            "\nHISTORY API",
            f"Status: {history_response.status_code}",
            pretty(
                {
                    "success": history_json["success"],
                    "count": history_json["count"],
                    "latest_history_id": history_json["history"][0]["id"],
                }
            ),
        ]
    )

    risk_response = client.post("/api/risk", json={"probability": 0.50})
    if risk_response.status_code != 200:
        raise RuntimeError("Risk API failed.")
    output_lines.extend(
        [
            "\nRISK API",
            f"Status: {risk_response.status_code}",
            pretty(risk_response.get_json()),
        ]
    )

    recommendation_response = client.post(
        "/api/recommendation",
        json={
            "bmi": 28.7,
            "glucose": 126.0,
            "smoking": "formerly smoked",
            "hypertension": True,
            "heart_disease": False,
            "age": 55,
            "risk_category": "Medium Risk",
        },
    )
    if recommendation_response.status_code != 200:
        raise RuntimeError("Recommendation API failed.")
    recommendation_json = recommendation_response.get_json()
    output_lines.extend(
        [
            "\nRECOMMENDATION API",
            f"Status: {recommendation_response.status_code}",
            pretty(
                {
                    "success": recommendation_json["success"],
                    "sections": list(
                        recommendation_json["recommendations"].keys()
                    ),
                }
            ),
        ]
    )

    explanation_response = client.get("/api/explainability")
    if explanation_response.status_code != 200:
        raise RuntimeError("Explainability API failed.")
    explanation_json = explanation_response.get_json()
    output_lines.extend(
        [
            "\nEXPLAINABILITY API",
            f"Status: {explanation_response.status_code}",
            pretty(
                {
                    "success": explanation_json["success"],
                    "scope": explanation_json["scope"],
                    "top_feature": explanation_json["top_features"][0],
                }
            ),
            "\nALL REQUIRED APIS PASSED",
        ]
    )

    output = "\n".join(output_lines) + "\n"
    print(output)
    REPORT_PATH.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
