"""Project-level configuration shim.

The real configuration lives in :mod:`app.config`.  This module re-exports it
so that tooling which expects a top-level ``config.py`` (for example
``flask --app config``) keeps working.
"""

from __future__ import annotations

from app.config import (  # noqa: F401
    ARTIFACTS_DIR,
    ASSETS_DIR,
    BASE_DIR,
    CONFIG_MAP,
    DATASET_DIR,
    ENCODERS_DIR,
    INSTANCE_DIR,
    MIGRATIONS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    BaseConfig,
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    get_config,
)

__all__ = [
    "ARTIFACTS_DIR",
    "ASSETS_DIR",
    "BASE_DIR",
    "CONFIG_MAP",
    "DATASET_DIR",
    "ENCODERS_DIR",
    "INSTANCE_DIR",
    "MIGRATIONS_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "BaseConfig",
    "DevelopmentConfig",
    "ProductionConfig",
    "TestingConfig",
    "get_config",
]
