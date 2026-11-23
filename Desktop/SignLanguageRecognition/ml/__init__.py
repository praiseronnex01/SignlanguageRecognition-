"""Machine-learning pipeline package for the MSL recognition system.

The modules in this package are deliberately independent of the Flask
application so that they can be executed from the command line::

    python -m ml.train --help
    python -m ml.collect_data --help
    python -m ml.compare_models --help
"""

__all__ = [
    "collect_data",
    "compare_models",
    "evaluate",
    "feature_engineering",
    "predict",
    "preprocess",
    "train",
]
