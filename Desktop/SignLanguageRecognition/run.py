"""Development entry point for the Malawi Sign Language Recognition System.

Run with::

    flask --app run.py run --debug
    python run.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from app import create_app  # noqa: E402

app = create_app(os.getenv("FLASK_CONFIG", "development"))


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") not in {"0", "false", "False", ""}
    print(f"Starting the MSL recognition system on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
