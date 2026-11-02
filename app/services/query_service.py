"""Services for querying experiments, models, and deployments from the database."""

from __future__ import annotations

from sqlalchemy.orm import Session, joinedload

from app.database.models import (
    Artifact,
    Deployment,
    Experiment,
    ExperimentStatus,
    Metric,
    ModelVersion,
)
from app.database.session import get_session


# ─── Experiment Service ───────────────────────────────────────────────────────


class ExperimentService:
    @staticmethod
    def get_all(session: Session, status: ExperimentStatus | None = None, limit: int = 100):
        """List experiments with optional status filter."""
        query = session.query(Experiment).options(
            joinedload(Experiment.metrics),
            joinedload(Experiment.model_version),
            joinedload(Experiment.artifacts),
        )
        if status:
            query = query.filter(Experiment.status == status)
        return query.order_by(Experiment.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_by_id(session: Session, experiment_id: str):
        """Get a single experiment by ID."""
        return (
            session.query(Experiment)
            .options(
                joinedload(Experiment.metrics),
                joinedload(Experiment.model_version),
                joinedload(Experiment.artifacts),
            )
            .filter(Experiment.id == experiment_id)
            .first()
        )

    @staticmethod
    def get_metrics(session: Session, experiment_id: str):
        """Get all metrics for an experiment."""
        return (
            session.query(Metric)
            .filter(Metric.experiment_id == experiment_id)
            .order_by(Metric.step, Metric.key, Metric.timestamp)
            .all()
        )

    @staticmethod
    def update_status(session: Session, experiment_id: str, status: ExperimentStatus):
        """Update an experiment's status."""
        exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            exp.status = status
            session.commit()
            return exp
        return None


# ─── Model Version Service ───────────────────────────────────────────────────


class ModelVersionService:
    @staticmethod
    def list_all(session: Session, model_name: str | None = None, limit: int = 50):
        """List registered model versions."""
        query = session.query(ModelVersion).options(
            joinedload(ModelVersion.experiment),
            joinedload(ModelVersion.deployments),
        )
        if model_name:
            query = query.filter(ModelVersion.model_name == model_name)
        return query.order_by(ModelVersion.model_name, ModelVersion.version.desc()).limit(limit).all()

    @staticmethod
    def get_latest(session: Session, model_name: str):
        """Get the latest version of a model."""
        return (
            session.query(ModelVersion)
            .filter(ModelVersion.model_name == model_name)
            .order_by(ModelVersion.version.desc())
            .first()
        )

    @staticmethod
    def get_by_id(session: Session, version_id: str):
        return (
            session.query(ModelVersion)
            .options(
                joinedload(ModelVersion.experiment),
                joinedload(ModelVersion.deployments),
            )
            .filter(ModelVersion.id == version_id)
            .first()
        )


# ─── Deployment Service ──────────────────────────────────────────────────────


class DeploymentService:
    @staticmethod
    def list_all(session: Session, environment: str | None = None):
        """List deployments with optional environment filter."""
        query = session.query(Deployment).options(
            joinedload(Deployment.model_version),
        )
        if environment:
            query = query.filter(Deployment.environment == environment)
        return query.order_by(Deployment.deployed_at.desc()).all()

    @staticmethod
    def get_by_id(session: Session, deployment_id: str):
        return (
            session.query(Deployment)
            .options(joinedload(Deployment.model_version))
            .filter(Deployment.id == deployment_id)
            .first()
        )
