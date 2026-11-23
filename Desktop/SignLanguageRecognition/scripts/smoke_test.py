"""Manual end-to-end smoke test over every registered route.

Run with::

    python scripts/smoke_test.py

It uses the Flask test client, so no server needs to be running.  The script
exercises the public pages, the authentication flow, the JSON API and the
role-based access control, and reports a pass/fail line for each check.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app import create_app  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label} {detail}")


def login(client, email: str, password: str) -> bool:
    response = client.post(
        "/auth/api/auth/login",
        json={"email": email, "password": password},
    )
    return response.status_code == 200


def main() -> int:
    app = create_app()
    app.config["WTF_CSRF_ENABLED"] = False

    # Unique suffix so the script can be executed repeatedly.
    stamp = str(int(time.time()))[-7:]
    tester_email = f"smoke-{stamp}@example.com"
    gesture_label = f"SMOKE_{stamp}"

    with app.test_client() as client:
        print("Public pages")
        for path in ("/", "/auth/login", "/auth/register"):
            response = client.get(path)
            check(f"GET {path}", response.status_code == 200, f"-> {response.status_code}")

        print("Anonymous access is redirected")
        for path in ("/dashboard", "/recognition", "/vocabulary", "/history", "/dataset", "/training", "/evaluation", "/admin"):
            response = client.get(path)
            check(
                f"GET {path} redirects when signed out",
                response.status_code in (302, 401, 403),
                f"-> {response.status_code}",
            )

        print("Authentication API")
        response = client.post(
            "/auth/api/auth/register",
            json={
                "full_name": "Smoke Tester",
                "email": tester_email,
                "password": "Smoke12345",
                "confirm_password": "Smoke12345",
            },
        )
        check("POST /api/auth/register", response.status_code == 201, f"-> {response.status_code}")

        response = client.post(
            "/auth/api/auth/login",
            json={"email": tester_email, "password": "wrong"},
        )
        check(
            "POST /api/auth/login rejects a bad password",
            response.status_code == 401,
            f"-> {response.status_code}",
        )

        response = client.post(
            "/auth/api/auth/login",
            json={"email": tester_email, "password": "Smoke12345"},
        )
        check("POST /api/auth/login", response.status_code == 200, f"-> {response.status_code}")

        response = client.get("/auth/api/auth/me")
        check("GET /api/auth/me", response.status_code == 200, f"-> {response.status_code}")

        print("Signed-in pages (standard user)")
        for path in (
            "/dashboard",
            "/recognition",
            "/vocabulary",
            "/history",
            "/auth/profile",
        ):
            response = client.get(path)
            check(f"GET {path}", response.status_code == 200, f"-> {response.status_code}")

        print("Administrator pages are protected")
        for path in ("/dataset", "/training", "/evaluation", "/admin"):
            response = client.get(path)
            check(
                f"GET {path} is forbidden for a standard user",
                response.status_code == 403,
                f"-> {response.status_code}",
            )

        print("JSON API (standard user)")
        for path in (
            "/api/dashboard/stats",
            "/vocabulary/api/vocabulary",
            "/recognition/api/status",
            "/history/api/history",
        ):
            response = client.get(path)
            check(f"GET {path}", response.status_code == 200, f"-> {response.status_code}")

        response = client.post("/vocabulary/api/vocabulary", json={"label": "X", "english_meaning": "Y"})
        check(
            "POST /api/vocabulary is forbidden for a standard user",
            response.status_code == 403,
            f"-> {response.status_code}",
        )

        print("Prediction endpoint")
        blank = io.BytesIO()
        import cv2
        import numpy as np

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.imwrite("blank.jpg", image)
        with open("blank.jpg", "rb") as handle:
            payload = handle.read()
        response = client.post(
            "/recognition/api/predict",
            data={"file": (io.BytesIO(payload), "frame.jpg")},
            content_type="multipart/form-data",
        )
        check("POST /api/recognition/predict", response.status_code == 200, f"-> {response.status_code}")
        body = response.get_json() or {}
        check(
            "the prediction response follows the documented contract",
            "success" in body and "recognized" in body and "hand_detected" in body,
            json.dumps(body)[:200],
        )
        check(
            "no gesture is reported when no model is published",
            body.get("recognized") is False,
            json.dumps(body)[:200],
        )

        response = client.post("/recognition/api/predict", data={"file": (io.BytesIO(b"nope"), "x.jpg")},
                               content_type="multipart/form-data")
        check("POST /api/recognition/predict rejects a broken image", response.status_code == 400,
              f"-> {response.status_code}")

        print("Administrator session")
        client.post("/auth/api/auth/logout")
        if not login(client, "admin@msl.local", "Admin12345"):
            check("administrator sign-in", False)
        else:
            check("administrator sign-in", True)

        for path in ("/dataset", "/training", "/evaluation", "/admin"):
            response = client.get(path)
            check(f"GET {path}", response.status_code == 200, f"-> {response.status_code}")

        print("Administrator API")
        for path in (
            "/dataset/api/dataset/stats",
            "/dataset/api/dataset/participants",
            "/dataset/api/dataset/samples",
            "/dataset/api/dataset/versions",
            "/training/api/training/runs",
            "/training/api/training/models",
            "/training/api/training/dataset-summary",
            "/evaluation/api/evaluation/latest",
            "/evaluation/api/evaluation/models",
            "/evaluation/api/evaluation/compare",
            "/admin/api/admin/users",
            "/admin/api/admin/audit-logs",
            "/admin/api/admin/overview",
        ):
            response = client.get(path)
            check(f"GET {path}", response.status_code == 200, f"-> {response.status_code}")

        print("Vocabulary administration")
        response = client.post(
            "/vocabulary/api/vocabulary",
            json={
                "label": gesture_label,
                "english_meaning": "Smoke test sign",
                "category": "Test",
                "hand_count": 1,
            },
        )
        check("POST /api/vocabulary", response.status_code == 201, f"-> {response.status_code}")
        gesture_id = (response.get_json() or {}).get("gesture", {}).get("id")

        if gesture_id:
            response = client.put(
                f"/vocabulary/api/vocabulary/{gesture_id}",
                json={"english_meaning": "Smoke test sign (updated)"},
            )
            check("PUT /api/vocabulary/<id>", response.status_code == 200, f"-> {response.status_code}")

            response = client.put(
                f"/vocabulary/api/vocabulary/{gesture_id}", json={"is_validated": True}
            )
            body = response.get_json() or {}
            check(
                "PUT /api/vocabulary/<id> can validate a sign",
                response.status_code == 200 and body.get("gesture", {}).get("is_validated") is True,
                f"-> {response.status_code}",
            )

            response = client.get(f"/vocabulary/api/vocabulary/{gesture_id}")
            check("GET /api/vocabulary/<id>", response.status_code == 200, f"-> {response.status_code}")

            response = client.delete(f"/vocabulary/api/vocabulary/{gesture_id}")
            check("DELETE /api/vocabulary/<id>", response.status_code == 200, f"-> {response.status_code}")

        print("Dataset administration")
        response = client.post(
            "/dataset/api/dataset/participants",
            json={"participant_code": "SMOKE01", "consent_status": "granted"},
        )
        check("POST /api/dataset/participants", response.status_code == 201, f"-> {response.status_code}")

        response = client.post(
            "/dataset/api/dataset/sessions", json={"participant_code": "SMOKE01"}
        )
        check("POST /api/dataset/sessions", response.status_code == 201, f"-> {response.status_code}")

        response = client.get("/api/dashboard/stats")
        body = response.get_json() or {}
        stats = body.get("stats", {})
        check(
            "the dashboard statistics come from the database",
            stats.get("gestures", {}).get("total", 0) > 0 and stats.get("users", {}).get("total", 0) > 0,
            json.dumps(stats)[:200],
        )

    if Path("blank.jpg").exists():
        Path("blank.jpg").unlink()

    print("-" * 60)
    print(f"Passed: {PASSED}   Failed: {FAILED}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
