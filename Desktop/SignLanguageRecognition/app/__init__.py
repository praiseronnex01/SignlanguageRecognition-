"""Flask application factory for the Malawi Sign Language Recognition System."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import get_config  # noqa: E402
from app.extensions import csrf, db, login_manager, migrate  # noqa: E402


def create_app(config_name: str | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        instance_relative_config=False,
        static_folder="static",
        template_folder="templates",
    )

    config_class = get_config(config_name)
    app.config.from_object(config_class)
    app.config["PROJECT_ROOT"] = str(BASE_DIR)

    if app.config.get("DEBUG"):
        app.logger.setLevel(logging.DEBUG)

    _configure_extensions(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _register_template_globals(app)
    _register_cli(app)

    if app.config.get("AUTO_MIGRATE_ON_START"):
        _prepare_database(app)
    else:
        with app.app_context():
            db.create_all()

    return app


def _configure_extensions(app: Flask) -> None:
    db.init_app(app)
    migrate.init_app(app, db, directory=str(BASE_DIR / "migrations"))
    login_manager.init_app(app)
    csrf.init_app(app)

    from app.models import AuditLog  # noqa: F401  (register models)

    @app.after_request
    def set_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response


def _register_blueprints(app: Flask) -> None:
    from app.blueprints.admin.routes import bp as admin_bp
    from app.blueprints.auth.routes import bp as auth_bp
    from app.blueprints.dashboard.routes import bp as dashboard_bp
    from app.blueprints.dataset.routes import bp as dataset_bp
    from app.blueprints.evaluation.routes import bp as evaluation_bp
    from app.blueprints.history.routes import bp as history_bp
    from app.blueprints.landing.routes import bp as landing_bp
    from app.blueprints.recognition.routes import bp as recognition_bp
    from app.blueprints.training.routes import bp as training_bp
    from app.blueprints.vocabulary.routes import bp as vocabulary_bp

    app.register_blueprint(landing_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(recognition_bp)
    app.register_blueprint(vocabulary_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(evaluation_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(admin_bp)


def _register_error_handlers(app: Flask) -> None:
    def wants_json() -> bool:
        if request.path.startswith("/api/"):
            return True
        accept = request.accept_mimetypes
        return accept.best == "application/json" and accept["application/json"] >= accept["text/html"]

    @app.errorhandler(400)
    def bad_request(error):
        if wants_json():
            return jsonify({"success": False, "error": "The request could not be processed."}), 400
        return render_template("errors/error.html", code=400, message="Bad request"), 400

    @app.errorhandler(401)
    def unauthorised(error):
        if wants_json():
            return jsonify({"success": False, "error": "Authentication is required."}), 401
        return render_template("errors/error.html", code=401, message="Authentication required"), 401

    @app.errorhandler(403)
    def forbidden(error):
        if wants_json():
            return jsonify(
                {"success": False, "error": "You do not have permission to perform this action."}
            ), 403
        return (
            render_template(
                "errors/error.html",
                code=403,
                message="You do not have permission to view this page.",
            ),
            403,
        )

    @app.errorhandler(404)
    def not_found(error):
        if wants_json():
            return jsonify({"success": False, "error": "The requested resource was not found."}), 404
        return render_template("errors/error.html", code=404, message="Page not found"), 404

    @app.errorhandler(413)
    def too_large(error):
        if wants_json():
            return jsonify(
                {"success": False, "error": "The uploaded file is too large."}
            ), 413
        return render_template("errors/error.html", code=413, message="Payload too large"), 413

    @app.errorhandler(429)
    def too_many(error):
        if wants_json():
            return jsonify({"success": False, "error": "Too many requests."}), 429
        return render_template("errors/error.html", code=429, message="Too many requests"), 429

    @app.errorhandler(500)
    def server_error(error):  # pragma: no cover - defensive
        db.session.rollback()
        app.logger.exception("Unhandled server error")
        if wants_json():
            return jsonify({"success": False, "error": "An internal server error occurred."}), 500
        return render_template("errors/error.html", code=500, message="Internal server error"), 500


def _register_template_globals(app: Flask) -> None:
    from app.models.gesture import Gesture

    @app.context_processor
    def inject_globals():
        return {
            "project_name": app.config.get("PROJECT_NAME"),
            "project_short_name": app.config.get("PROJECT_SHORT_NAME"),
            "msl_reference_url": app.config.get("MSL_REFERENCE_URL"),
            "current_year": __import__("datetime").datetime.now().year,
        }

    @app.template_filter("datetime")
    def format_datetime(value, fmt: str = "%d %b %Y, %H:%M"):
        if value is None:
            return "-"
        try:
            return value.strftime(fmt)
        except AttributeError:
            return str(value)

    @app.template_filter("filesize")
    def format_filesize(value):
        try:
            size = float(value)
        except (TypeError, ValueError):
            return "-"
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"


def _register_cli(app: Flask) -> None:
    @app.cli.command("create-admin")
    def create_admin_command():  # pragma: no cover - CLI helper
        """Create an administrator account."""
        from scripts.create_admin import create_admin

        create_admin()

    @app.cli.command("seed-vocabulary")
    def seed_vocabulary_command():  # pragma: no cover - CLI helper
        """Load the candidate MSL vocabulary into the database."""
        from scripts.seed_vocabulary import seed_vocabulary

        count = seed_vocabulary()
        print(f"Seeded {count} candidate gestures.")

    @app.cli.command("check-environment")
    def check_environment_command():  # pragma: no cover - CLI helper
        """Report the state of the local environment."""
        from scripts.check_environment import run_checks

        run_checks()


def _prepare_database(app: Flask) -> None:
    """Bring the schema up to date.

    The step is skipped while the ``flask db ...`` CLI commands are running so
    that ``flask db migrate`` always compares the models against the real
    database state.
    """
    if os.environ.get("FLASK_RUN_FROM_CLI", "").lower() in {"1", "true", "yes"}:
        return

    migrations_env = BASE_DIR / "migrations" / "env.py"
    if not migrations_env.exists():
        app.logger.warning(
            "Migrations have not been initialised. Run 'flask db init' followed "
            "by 'flask db upgrade'."
        )
        return

    with app.app_context():
        try:
            from flask_migrate import upgrade

            upgrade(directory=str(BASE_DIR / "migrations"))
            app.logger.info("Database schema upgraded via migrations.")
        except Exception as exc:  # pragma: no cover - environment specific
            app.logger.error(
                "Could not upgrade the database schema (%s). Run 'flask db upgrade' "
                "manually before using the application.",
                exc,
            )
