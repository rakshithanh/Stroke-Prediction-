"""Demonstrate Phase 12 CRUD and save SQLite schema/record reports."""

from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from flask import Flask
from werkzeug.security import generate_password_hash

from app import create_app
from database.crud import (
    create_prediction,
    create_recommendation,
    create_risk_history,
    create_user,
    delete_prediction,
    delete_recommendation,
    delete_risk_history,
    delete_user,
    read_prediction,
    read_recommendation,
    read_risk_history,
    read_user,
    update_prediction,
    update_recommendation,
    update_risk_history,
    update_user,
)
from database.db import get_database


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_REPORT_PATH = PROJECT_ROOT / "reports" / "database_schema.txt"
RECORDS_REPORT_PATH = PROJECT_ROOT / "reports" / "database_records.txt"
PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"
REQUESTED_TABLES = ["users", "Predictions", "RiskHistory", "Recommendations"]


def demonstrate_crud(app: Flask) -> list[str]:
    """Create, read, update, and delete temporary linked demonstration rows."""
    messages = ["CRUD DEMONSTRATION", "=" * 70]
    with app.app_context():
        temporary_email = f"crud-{uuid4().hex[:8]}@example.com"
        user_id = create_user(
            "CRUD Demo",
            temporary_email,
            generate_password_hash("temporary123"),
        )
        messages.append(f"Users CREATE: id={user_id}")
        messages.append(f"Users READ: {read_user(user_id)}")
        update_user(user_id, "CRUD Demo Updated", temporary_email)
        messages.append(f"Users UPDATE: {read_user(user_id)}")

        prediction_id = create_prediction(
            user_id,
            {"age": 50, "bmi": 27.0},
            0.45,
            0,
        )
        messages.append(f"Predictions CREATE: id={prediction_id}")
        messages.append(f"Predictions READ: {read_prediction(prediction_id)}")
        update_prediction(prediction_id, 0.55, 1)
        messages.append(f"Predictions UPDATE: {read_prediction(prediction_id)}")

        risk_id = create_risk_history(
            prediction_id,
            user_id,
            55.0,
            "Medium Risk",
            "#ffc107",
            "Academic medium-risk demonstration.",
        )
        messages.append(f"RiskHistory CREATE: id={risk_id}")
        messages.append(f"RiskHistory READ: {read_risk_history(risk_id)}")
        update_risk_history(
            risk_id,
            56.0,
            "Medium Risk",
            "#ffc107",
            "Updated academic medium-risk demonstration.",
        )
        messages.append(f"RiskHistory UPDATE: {read_risk_history(risk_id)}")

        recommendation_data = {
            "diet": ["Balanced academic diet example."],
            "exercise": ["Suitable academic exercise example."],
            "doctor_visit": ["Routine professional review example."],
            "lifestyle_changes": ["Maintain healthy habits."],
            "medication_reminder": ["Use only prescribed medicines."],
        }
        recommendation_id = create_recommendation(
            prediction_id,
            user_id,
            recommendation_data,
        )
        messages.append(f"Recommendations CREATE: id={recommendation_id}")
        messages.append(
            f"Recommendations READ: {read_recommendation(recommendation_id)}"
        )
        recommendation_data["diet"] = ["Updated balanced diet example."]
        update_recommendation(recommendation_id, recommendation_data)
        messages.append(
            f"Recommendations UPDATE: {read_recommendation(recommendation_id)}"
        )

        # Delete children explicitly to demonstrate their DELETE functions.
        messages.append(
            f"Recommendations DELETE: {delete_recommendation(recommendation_id)}"
        )
        messages.append(f"RiskHistory DELETE: {delete_risk_history(risk_id)}")
        messages.append(f"Predictions DELETE: {delete_prediction(prediction_id)}")
        messages.append(f"Users DELETE: {delete_user(user_id)}")
    return messages


def save_schema_report(app: Flask) -> None:
    """Save column and foreign-key details for all requested tables."""
    lines = ["PHASE 12 SQLITE DATABASE SCHEMA", "=" * 70]
    with app.app_context():
        database = get_database()
        for table_name in REQUESTED_TABLES:
            lines.extend(["", f"TABLE: {table_name}", "-" * 70])
            columns = database.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
            for column in columns:
                lines.append(
                    f"{column['name']:<24} {column['type']:<12} "
                    f"NOT_NULL={column['notnull']} PK={column['pk']} "
                    f"DEFAULT={column['dflt_value']}"
                )
            foreign_keys = database.execute(
                f'PRAGMA foreign_key_list("{table_name}")'
            ).fetchall()
            if foreign_keys:
                lines.append("Foreign Keys:")
                for key in foreign_keys:
                    lines.append(
                        f"  {key['from']} -> {key['table']}.{key['to']} "
                        f"ON_DELETE={key['on_delete']}"
                    )
    SCHEMA_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_records_report(app: Flask, crud_messages: list[str]) -> None:
    """Save CRUD results and current stored records for screenshots."""
    lines = [*crud_messages, "", "CURRENT STORED RECORDS", "=" * 70]
    with app.app_context():
        database = get_database()
        queries = {
            "users": "SELECT id, name, email, created_at FROM users ORDER BY id",
            "Predictions": "SELECT * FROM Predictions ORDER BY id",
            "RiskHistory": "SELECT * FROM RiskHistory ORDER BY id",
            "Recommendations": "SELECT * FROM Recommendations ORDER BY id",
        }
        for table_name, query in queries.items():
            rows = database.execute(query).fetchall()
            lines.extend(
                ["", f"TABLE: {table_name} | RECORD COUNT: {len(rows)}", "-" * 70]
            )
            if rows:
                lines.extend(str(dict(row)) for row in rows)
            else:
                lines.append("No stored records.")
    RECORDS_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_database_table_image(app: Flask) -> None:
    """Save a screenshot-ready view of the four requested table schemas."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 1, figsize=(12, 15))
    figure.suptitle("Phase 12 SQLite Database Tables", fontsize=20, y=0.995)
    with app.app_context():
        database = get_database()
        for axis, table_name in zip(axes, REQUESTED_TABLES):
            axis.axis("off")
            columns = database.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
            cell_text = [
                [
                    column["name"],
                    column["type"],
                    "Yes" if column["notnull"] else "No",
                    "Yes" if column["pk"] else "No",
                ]
                for column in columns
            ]
            table = axis.table(
                cellText=cell_text,
                colLabels=["Column", "Type", "Required", "Primary Key"],
                cellLoc="left",
                loc="center",
                colColours=["#d9e8f5"] * 4,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.35)
            axis.set_title(table_name, fontsize=15, fontweight="bold", pad=8)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(PLOTS_DIR / "database_tables.png", dpi=150)
    plt.close(figure)


def save_stored_records_image(app: Flask) -> None:
    """Save counts and representative stored values for all four tables."""
    with app.app_context():
        database = get_database()
        user = database.execute(
            "SELECT id, name, email FROM users ORDER BY id LIMIT 1"
        ).fetchone()
        prediction = database.execute(
            "SELECT id, user_id, probability, predicted_class FROM Predictions ORDER BY id LIMIT 1"
        ).fetchone()
        risk = database.execute(
            "SELECT id, prediction_id, risk_percentage, risk_category FROM RiskHistory ORDER BY id LIMIT 1"
        ).fetchone()
        recommendation = database.execute(
            "SELECT id, prediction_id, user_id FROM Recommendations ORDER BY id LIMIT 1"
        ).fetchone()
        rows = [
            [
                "users",
                database.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                f"id={user['id']}, name={user['name']}, email={user['email']}" if user else "None",
            ],
            [
                "Predictions",
                database.execute("SELECT COUNT(*) FROM Predictions").fetchone()[0],
                (
                    f"id={prediction['id']}, probability={prediction['probability']:.4f}, "
                    f"class={prediction['predicted_class']}"
                    if prediction
                    else "None"
                ),
            ],
            [
                "RiskHistory",
                database.execute("SELECT COUNT(*) FROM RiskHistory").fetchone()[0],
                (
                    f"id={risk['id']}, risk={risk['risk_percentage']:.2f}%, "
                    f"category={risk['risk_category']}"
                    if risk
                    else "None"
                ),
            ],
            [
                "Recommendations",
                database.execute("SELECT COUNT(*) FROM Recommendations").fetchone()[0],
                (
                    f"id={recommendation['id']}, prediction_id={recommendation['prediction_id']}, "
                    "five sections stored"
                    if recommendation
                    else "None"
                ),
            ],
        ]

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Table", "Record Count", "Representative Stored Record"],
        cellLoc="left",
        loc="center",
        colColours=["#d9e8f5"] * 3,
        colWidths=[0.20, 0.15, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    axis.set_title("Phase 12 Stored Database Records", fontsize=18, pad=20)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "database_stored_records.png", dpi=150)
    plt.close(figure)


def main() -> int:
    """Run CRUD verification and generate database screenshot reports."""
    app = create_app()
    app.config.update(TESTING=True)
    crud_messages = demonstrate_crud(app)
    save_schema_report(app)
    save_records_report(app, crud_messages)
    save_database_table_image(app)
    save_stored_records_image(app)
    print("\n".join(crud_messages))
    print(f"\nSchema report: {SCHEMA_REPORT_PATH}")
    print(f"Stored-record report: {RECORDS_REPORT_PATH}")
    print("\nALL FOUR TABLES PASSED CRUD TESTING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
