"""SQLAlchemy model package.

Importing every model here guarantees that Flask-Migrate's autogenerate and
the ``db.create_all()`` fallback see the complete schema.
"""

from app.models.audit_log import AuditLog
from app.models.collection_session import CollectionSession
from app.models.dataset_sample import DatasetSample
from app.models.evaluation_metric import EvaluationMetric
from app.models.gesture import Gesture
from app.models.model_version import ModelVersion
from app.models.participant import Participant
from app.models.recognition_history import RecognitionHistory
from app.models.training_run import TrainingRun
from app.models.user import User

__all__ = [
    "AuditLog",
    "CollectionSession",
    "DatasetSample",
    "EvaluationMetric",
    "Gesture",
    "ModelVersion",
    "Participant",
    "RecognitionHistory",
    "TrainingRun",
    "User",
]
