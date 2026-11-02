"""SQLAlchemy database models for experiment tracking, model registry, and deployment history."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlgorithmType(str, enum.Enum):
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"


class DeploymentStatus(str, enum.Enum):
    PENDING = "pending"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


# ─── Models ───────────────────────────────────────────────────────────────────


class Experiment(Base):
    """Tracks a single training experiment."""

    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    algorithm = Column(Enum(AlgorithmType), nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING)
    config_json = Column(Text, nullable=True)  # JSON-encoded hyperparameters
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    metrics = relationship("Metric", back_populates="experiment", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="experiment", cascade="all, delete-orphan")
    model_version = relationship("ModelVersion", back_populates="experiment", uselist=False)

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name}, status={self.status})>"


class Metric(Base):
    """Per-step training or evaluation metrics."""

    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    step = Column(Integer, nullable=False)
    key = Column(String(255), nullable=False)  # e.g., "train/loss", "val/accuracy"
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="metrics")

    __table_args__ = (
        # One metric key per step per experiment should be unique
        # We allow multiple keys per step (e.g., loss + accuracy)
    )

    def __repr__(self) -> str:
        return f"<Metric(exp={self.experiment_id}, step={self.step}, {self.key}={self.value})>"


class ModelVersion(Base):
    """A registered model version tied to an experiment."""

    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False, unique=True)
    model_name = Column(String(255), nullable=False)
    version = Column(Integer, nullable=False)  # Monotonically increasing
    algorithm = Column(Enum(AlgorithmType), nullable=False)
    metrics_json = Column(Text, nullable=True)  # Final evaluation metrics as JSON
    model_uri = Column(String(1024), nullable=False)  # Path or S3 URI to model artifact
    tags_json = Column(Text, nullable=True)  # Tags like "best", "production"
    metadata_json = Column(Text, nullable=True)
    registered_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="model_version")
    deployments = relationship("Deployment", back_populates="model_version", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ModelVersion(name={self.model_name}, version={self.version})>"


class Artifact(Base):
    """Model checkpoint files or training artifacts."""

    __tablename__ = "artifacts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    name = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    artifact_type = Column(String(64), default="checkpoint")  # checkpoint, config, data split, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="artifacts")

    def __repr__(self) -> str:
        return f"<Artifact(name={self.name}, type={self.artifact_type})>"


class Deployment(Base):
    """Model deployment history."""

    __tablename__ = "deployments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(String(36), ForeignKey("model_versions.id"), nullable=False)
    environment = Column(String(64), nullable=False)  # staging, production, canary
    status = Column(Enum(DeploymentStatus), default=DeploymentStatus.PENDING)
    endpoint_url = Column(String(1024), nullable=True)
    deployed_by = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    deployed_at = Column(DateTime, default=datetime.utcnow)
    undeployed_at = Column(DateTime, nullable=True)

    model_version = relationship("ModelVersion", back_populates="deployments")

    def __repr__(self) -> str:
        return f"<Deployment(model={self.model_version_id}, env={self.environment}, status={self.status})>"


def init_db(engine_url: str | None = None) -> None:
    """Initialize all tables. Creates the database if it doesn't exist."""
    from app.database.session import get_engine

    if engine_url:
        engine = create_engine(engine_url)
    else:
        engine = get_engine()

    Base.metadata.create_all(engine)
