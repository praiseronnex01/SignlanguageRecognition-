"""User account model and role definitions."""

from __future__ import annotations

from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(db.Model):
    """An application account.

    ``role`` drives coarse-grained authorisation: ``admin`` users may manage
    the vocabulary, dataset, training runs and other accounts; ``user``
    accounts may run live recognition and inspect their own history.
    """

    __tablename__ = "users"

    ROLE_ADMIN = "admin"
    ROLE_USER = "user"
    VALID_ROLES = {ROLE_ADMIN, ROLE_USER}

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default=ROLE_USER, index=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    # --- Flask-Login interface (implemented explicitly to avoid clashing
    # --- with the ``is_active`` database column). -------------------------
    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def is_anonymous(self) -> bool:
        return False

    def get_id(self) -> str:
        return str(self.id)

    # --- Password handling ------------------------------------------------
    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        if not self.password_hash or not raw_password:
            return False
        return check_password_hash(self.password_hash, raw_password)

    # --- Roles ------------------------------------------------------------
    @property
    def is_admin(self) -> bool:
        return self.role == self.ROLE_ADMIN

    def to_dict(self, include_email: bool = True) -> dict:
        data = {
            "id": self.id,
            "full_name": self.full_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_email:
            data["email"] = self.email
        return data

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<User {self.id} {self.email!r} role={self.role!r}>"
