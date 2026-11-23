"""Public landing page."""

from __future__ import annotations

from flask import Blueprint, render_template

from app.models.gesture import Gesture

bp = Blueprint("landing", __name__)


@bp.route("/")
def index():
    """The public entry point of the application."""
    validated_count = Gesture.query.filter_by(is_validated=True, is_active=True).count()
    return render_template(
        "landing.html",
        validated_count=validated_count,
    )
