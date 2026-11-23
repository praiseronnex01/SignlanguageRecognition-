"""Create an administrator account from the command line.

Usage::

    python scripts/create_admin.py
    python scripts/create_admin.py --email admin@example.com --password "S3cret!" --name "Admin"
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models.user import User  # noqa: E402


def create_admin(
    email: str | None = None,
    password: str | None = None,
    full_name: str | None = None,
    interactive: bool = True,
) -> User:
    """Create (or promote) an administrator account."""
    app = create_app()
    with app.app_context():
        if interactive:
            email = email or input("Administrator email: ").strip()
            full_name = full_name or input("Full name: ").strip()
            if not password:
                password = getpass.getpass("Password: ")
                confirmation = getpass.getpass("Confirm password: ")
                if password != confirmation:
                    raise SystemExit("The passwords do not match.")

        if not email or not password:
            raise SystemExit("An email address and a password are required.")
        if len(password) < 8:
            raise SystemExit("The password must be at least eight characters long.")

        normalised = email.strip().lower()
        user = User.query.filter_by(email=normalised).first()
        if user is None:
            user = User(
                email=normalised,
                full_name=(full_name or "Administrator").strip(),
                role=User.ROLE_ADMIN,
            )
            user.set_password(password)
            db.session.add(user)
            action = "created"
        else:
            user.role = User.ROLE_ADMIN
            user.is_active = True
            if full_name:
                user.full_name = full_name.strip()
            user.set_password(password)
            action = "updated"

        db.session.commit()
        print(f"Administrator {action}: {user.email}")
        return user


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an administrator account.")
    parser.add_argument("--email", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--name", dest="full_name", default=None)
    args = parser.parse_args()

    interactive = not (args.email and args.password)
    try:
        create_admin(
            email=args.email,
            password=args.password,
            full_name=args.full_name,
            interactive=interactive,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
