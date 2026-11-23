"""Seed the candidate MSL vocabulary into the database.

The candidate entries below were taken from the **Malawian Sign Language
Dictionary** published by MANAD (https://msld.manadmw.org/).  For every entry
the dictionary's own metadata is reproduced:

* the dictionary entry number and the English gloss,
* the Chichewa term(s) for the sign,
* the dictionary's *handshape*, *location* and *movement* attribute codes,
* the number of hands used by the sign,
* a direct link to the dictionary entry.

Why these particular signs?
---------------------------
A webcam-only system observes the **hand**, nothing else.  The place where a
sign is performed relative to the body (head, chest, shoulder ...) is
therefore invisible to the camera, and the movement component of a sign is
invisible to a static-landmark classifier.  Two MSL signs are consequently
only separable when their *handshape signature* differs.

The candidates below were selected by analysing the handshape metadata of the
MSL dictionary and keeping the signs whose handshape signature is **unique**
inside the vocabulary.  Signs such as "black" (handshape ``p``) are therefore
deliberately *not* seeded, because the dictionary contains many other signs
with the same handshape (``hate``, ``purple``, ``turkey``, ``football`` ...)
which a static-landmark classifier cannot tell apart.

**No entry is marked as validated by this script.**  Validation is an explicit
decision by a competent Malawi Sign Language signer or a knowledgeable
instructor and must be recorded in the Vocabulary interface before the sign
may be used for recognition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models.gesture import (  # noqa: E402
    HAND_ONE,
    HAND_TWO,
    STATUS_CANDIDATE,
    Gesture,
)

DICTIONARY_URL = "https://msld.manadmw.org/"

#: Human readable explanation of the dictionary attribute codes.
LOCATION_LABELS = {
    "head": "at the head",
    "chest_heart_stomach": "at the chest",
    "arm_elbow": "at the arm/elbow",
    "upperarm_shoulder": "at the upper arm/shoulder",
    "hand_wrist_palm_finger_nail": "at the hand/wrist/palm",
    "nsh_nsm_nsl": "in neutral space",
}

MOVEMENT_LABELS = {
    "str_arc_pa_wav": "straight / arc / palm / wave",
    "turn": "turning",
    "cir": "circular",
    "op_cl": "opening and closing",
    "wa": "wrist action",
    "wi": "wrist movement",
    "be": "bending",
    "nm_nhm": "near-movement / no-hand-movement",
}

#: Candidates taken from the Malawian Sign Language Dictionary.
#:
#: ``handshape``/``location``/``movement`` are the dictionary's own attribute
#: codes.  ``hands`` is the number of hands the dictionary records for the
#: sign and selects the feature schema (1 -> 63 features, 2 -> 128 features).
CANDIDATES = [
    {
        "label": "FEELING",
        "english_meaning": "Feeling",
        "category": "Emotions",
        "chichewa": "-mva, khudza, -mvetsa",
        "number": 125,
        "hands": 1,
        "handshape": ["fi"],
        "location": "chest_heart_stomach",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "SURPRISE",
        "english_meaning": "Surprise",
        "category": "Emotions",
        "chichewa": "-dabwitsa, dodometsa",
        "number": 134,
        "hands": 1,
        "handshape": ["xi"],
        "location": "head",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "MONDAY",
        "english_meaning": "Monday",
        "category": "Time",
        "chichewa": "lolemba",
        "number": 153,
        "hands": 1,
        "handshape": ["m"],
        "location": "nsh_nsm_nsl",
        "movement": ["cir"],
    },
    {
        "label": "WEDNESDAY",
        "english_meaning": "Wednesday",
        "category": "Time",
        "chichewa": "lachitatu",
        "number": 155,
        "hands": 1,
        "handshape": ["w"],
        "location": "nsh_nsm_nsl",
        "movement": ["cir"],
    },
    {
        "label": "KNOWLEDGE",
        "english_meaning": "Knowledge / intelligent",
        "category": "Education",
        "chichewa": "chidziwitso, nzeru",
        "number": 202,
        "hands": 1,
        "handshape": ["vc"],
        "location": "head",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "BEER",
        "english_meaning": "Beer",
        "category": "Food and Drink",
        "chichewa": "beer",
        "number": 337,
        "hands": 1,
        "handshape": ["gc"],
        "location": "nsh_nsm_nsl",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "KASUNGU",
        "english_meaning": "Kasungu / tobacco",
        "category": "Places",
        "chichewa": "Kasungu, Fodya",
        "number": 359,
        "hands": 1,
        "handshape": ["k"],
        "location": "head",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "EXPLAIN",
        "english_meaning": "Explain / advise",
        "category": "Basic Conversation",
        "chichewa": "not separately recorded in the dictionary entry",
        "number": 44,
        "hands": 2,
        "handshape": ["o", "bc"],
        "location": "nsh_nsm_nsl",
        "movement": ["str_arc_pa_wav", "op_cl"],
    },
    {
        "label": "DAUGHTER",
        "english_meaning": "Daughter",
        "category": "Family and people",
        "chichewa": "mwana wamkazi",
        "number": 94,
        "hands": 2,
        "handshape": ["go", "bc"],
        "location": "nsh_nsm_nsl",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "THIEF",
        "english_meaning": "Thief / steal",
        "category": "Family and people",
        "chichewa": "mbava, wakuba, mbala, tsizinamtole, kuba",
        "number": 107,
        "hands": 2,
        "handshape": ["v", "s"],
        "location": "nsh_nsm_nsl",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "JULY",
        "english_meaning": "July",
        "category": "Time",
        "chichewa": "Julayi",
        "number": 146,
        "hands": 2,
        "handshape": ["i", "y"],
        "location": "nsh_nsm_nsl",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "DAY",
        "english_meaning": "Day",
        "category": "Time",
        "chichewa": "tsiku",
        "number": 152,
        "hands": 2,
        "handshape": ["z", "b"],
        "location": "nsh_nsm_nsl",
        "movement": ["str_arc_pa_wav"],
    },
    {
        "label": "COUNTRY",
        "english_meaning": "Country",
        "category": "Places",
        "chichewa": "dziko",
        "number": 347,
        "hands": 2,
        "handshape": ["c", "s"],
        "location": "arm_elbow",
        "movement": ["cir"],
    },
    {
        "label": "VILLAGE",
        "english_meaning": "Village / grass",
        "category": "Places",
        "chichewa": "mudzi, udzu",
        "number": 349,
        "hands": 2,
        "handshape": ["t", "s"],
        "location": "nsh_nsm_nsl",
        "movement": ["cir"],
    },
    {
        "label": "BLANTYRE",
        "english_meaning": "Blantyre",
        "category": "Places",
        "chichewa": "Blantyre",
        "number": 361,
        "hands": 2,
        "handshape": ["b", "t"],
        "location": "nsh_nsm_nsl",
        "movement": [],
    },
]

#: Sign numbers whose dictionary slug is used to build the source URL.
SLUGS = {
    44: "explain-advise",
    83: "mother",
    88: "brother",
    89: "sister",
    94: "daughter",
    107: "thief-steal",
    125: "feeling",
    134: "surprise",
    146: "july",
    152: "day",
    153: "monday",
    155: "wednesday",
    202: "knowledge-intelligent",
    337: "beer",
    347: "country",
    349: "village-grass",
    359: "kasungu-tobacco",
    361: "blantyre",
}


def build_description(candidate: dict) -> str:
    """Compose the description from the dictionary metadata."""
    hands = candidate["hands"]
    handshape = ", ".join(candidate["handshape"])
    location = LOCATION_LABELS.get(candidate["location"], candidate["location"])
    movement = candidate["movement"]
    movement_text = (
        ", ".join(MOVEMENT_LABELS.get(code, code) for code in movement)
        if movement
        else "no movement component recorded"
    )

    lines = [
        f"Malawian Sign Language Dictionary entry #{candidate['number']} "
        f"({candidate['chichewa']}).",
        f"Performed with {hands} hand{'s' if hands > 1 else ''}; "
        f"dictionary handshape '{handshape}' {location}; "
        f"movement: {movement_text}.",
        "A webcam-only classifier observes the handshape and hand configuration "
        "but not the location relative to the body nor the movement, so only the "
        "static handshape signature of this sign is modelled.",
        "STATUS: candidate only. A competent Malawi Sign Language signer must "
        "confirm the performance of this sign before it is marked as validated.",
    ]
    return " ".join(lines)


def seed_vocabulary(overwrite: bool = False, verbose: bool = True) -> int:
    """Insert the candidate vocabulary.  Returns the number of rows created."""
    app = create_app()
    created = 0
    with app.app_context():
        for candidate in CANDIDATES:
            slug = SLUGS.get(candidate["number"], candidate["label"].lower())
            source = f"{DICTIONARY_URL}sign/{candidate['number']}-{slug}/"

            existing = Gesture.query.filter_by(label=candidate["label"]).first()
            if existing is not None:
                if not overwrite:
                    continue
                existing.english_meaning = candidate["english_meaning"]
                existing.category = candidate["category"]
                existing.description = build_description(candidate)
                existing.hand_count = (
                    HAND_TWO if candidate["hands"] == 2 else HAND_ONE
                )
                existing.source_reference = source
                existing.validation_notes = (
                    "Seeded from the Malawian Sign Language Dictionary; "
                    "awaiting verification by an MSL signer."
                )
                db.session.commit()
                continue

            gesture = Gesture(
                label=candidate["label"],
                english_meaning=candidate["english_meaning"],
                category=candidate["category"],
                description=build_description(candidate),
                hand_count=HAND_TWO if candidate["hands"] == 2 else HAND_ONE,
                is_validated=False,
                is_active=True,
                validation_status=STATUS_CANDIDATE,
                validation_notes=(
                    "Seeded from the Malawian Sign Language Dictionary; "
                    "awaiting verification by an MSL signer."
                ),
                source_reference=source,
            )
            db.session.add(gesture)
            created += 1

        db.session.commit()

    if verbose:
        print(f"Seeded {created} candidate gesture(s) from the MSL dictionary.")
        print(
            "All seeded entries are CANDIDATES. Sign in as an administrator and "
            "use Vocabulary -> 'Mark validated' after a competent MSL signer has "
            "verified each sign."
        )
    return created


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed the candidate MSL vocabulary from the MSL dictionary."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Update existing entries instead of skipping them.",
    )
    args = parser.parse_args()
    seed_vocabulary(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
