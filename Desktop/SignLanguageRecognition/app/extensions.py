"""Flask extension singletons.

Kept in their own module so that blueprints, services and the application
factory can import them without creating circular dependencies.
"""

from flask import flash, jsonify, redirect, request, url_for
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()

login_manager.login_view = "auth.login_page"
login_manager.login_message = "Please sign in to continue."
login_manager.login_message_category = "warning"
login_manager.session_protection = "basic"


@login_manager.unauthorized_handler
def handle_unauthorized():
    if "/api/" in request.path:
        return jsonify(success=False, error=login_manager.login_message), 401
    flash(login_manager.login_message, login_manager.login_message_category)
    return redirect(url_for(login_manager.login_view, next=request.full_path))


@login_manager.user_loader
def load_user(user_id: str):
    """Rehydrate the logged-in user for each request."""
    from app.models.user import User

    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None
