"""Basic configuration for the academic Flask application."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Simple development configuration used during the dissertation project."""

    SECRET_KEY = "academic-project-development-key"
    DATABASE_PATH = BASE_DIR / "database" / "stroke_prediction.db"
