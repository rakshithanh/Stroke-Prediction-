"""Development entry point for the Phase 1 Flask application."""

from app import create_app


app = create_app()


if __name__ == "__main__":
    # Debug mode is convenient for local academic demonstrations only.
    # Disable the reloader so one predictable process is used for screenshots.
    app.run(debug=True, use_reloader=False)
