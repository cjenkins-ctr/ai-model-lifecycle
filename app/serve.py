"""FastAPI application for the model registry and experiment API."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import AlgorithmType, ExperimentStatus, DeploymentStatus, init_db
from app.database.session import get_session
from app.logging_config import setup_logging, get_logger
from app.pipeline.orchestrator import PipelineContext, run_pipeline
from app.services.query_service import (
    ExperimentService,
    ModelVersionService,
    DeploymentService,
)

setup_logging()
logger = get_logger(__name__)

# ─── Main Entry ───────────────────────────────────────────────────────────────


def main():
    """CLI entry point for model-serve command."""
    from app.cli.main import serve as _serve
    _serve()

# ─── FastAPI App ──────────────────────────────────────────────────────────────

init_db()

app = FastAPI(
    title="AI Model Lifecycle API",
    description="Build, train, and deploy models with experiment tracking and model registry.",
    version="0.1.0",
)


# ─── Pydantic Models ─────────────────────────────────────────────────────────


class ExperimentCreate(BaseModel):
    name: str = Field(..., description="Experiment name")
    algorithm: str = Field(..., description="Algorithm type (sklearn, pytorch, tensorflow, transformers)")
    config: dict = Field(default_factory=dict, description="Hyperparameters and training config")


class ExperimentResponse(BaseModel):
    id: str
    name: str
    algorithm: str
    status: str
    config_json: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class MetricLog(BaseModel):
    key: str = Field(..., description="Metric name (e.g., 'train/loss')")
    value: float = Field(..., description="Metric value")
    step: int = Field(default=0, description="Training step or epoch")


class ModelRegister(BaseModel):
    model_name: str = Field(..., description="Model name")
    model_uri: str = Field(..., description="Path or URI to model artifact")
    tags: list[str] = Field(default_factory=list, description="Tags for the model version")


class ModelDeploy(BaseModel):
    environment: str = Field(default="staging", description="Target environment")
    deployed_by: Optional[str] = None
    notes: Optional[str] = None


# ─── Helper ───────────────────────────────────────────────────────────────────


def _get_session():
    """Dependency: get a database session."""
    from app.database.session import get_session as _gs
    with _gs() as session:
        yield session


# ─── Experiment Endpoints ─────────────────────────────────────────────────────


@app.post("/api/v1/experiments", response_model=ExperimentResponse, status_code=201)
def create_experiment(experiment_data: ExperimentCreate, session: Session = Depends(_get_session)):
    """Create a new experiment."""
    try:
        algo = AlgorithmType(experiment_data.algorithm)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid algorithm: {experiment_data.algorithm}. Choose from: sklearn, pytorch, tensorflow, transformers")

    from app.pipeline.orchestrator import stage_build, PipelineContext

    ctx = PipelineContext(
        experiment_id=str(uuid.uuid4()),
        algorithm=algo,
        config=experiment_data.config,
        session=session,
    )
    ctx = stage_build(ctx)

    return ExperimentResponse(
        id=ctx.experiment_id,
        name=ctx.config.get("model_name", experiment_data.name),
        algorithm=algo.value,
        status="pending",
        config_json=experiment_data.config,
    )


@app.get("/api/v1/experiments")
def list_experiments(status: Optional[str] = None, limit: int = 100, session: Session = Depends(_get_session)):
    """List all experiments."""
    filter_status = ExperimentStatus(status) if status else None
    experiments = ExperimentService.get_all(session, status=filter_status, limit=limit)
    return [
        {
            "id": e.id,
            "name": e.name,
            "algorithm": e.algorithm.value,
            "status": e.status.value,
            "created_at": e.created_at.isoformat() if e.created_at else None,
            "completed_at": e.completed_at.isoformat() if e.completed_at else None,
        }
        for e in experiments
    ]


@app.get("/api/v1/experiments/{experiment_id}")
def get_experiment(experiment_id: str, session: Session = Depends(_get_session)):
    """Get experiment details."""
    exp = ExperimentService.get_by_id(session, experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    metrics = ExperimentService.get_metrics(session, experiment_id)

    return {
        "id": exp.id,
        "name": exp.name,
        "algorithm": exp.algorithm.value,
        "status": exp.status.value,
        "config_json": json.loads(exp.config_json) if exp.config_json else None,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        "error_message": exp.error_message,
        "metrics": [
            {"step": m.step, "key": m.key, "value": m.value, "timestamp": m.timestamp.isoformat()}
            for m in metrics
        ],
    }


@app.post("/api/v1/experiments/{experiment_id}/metrics")
def log_metric(experiment_id: str, metric_data: MetricLog, session: Session = Depends(_get_session)):
    """Log a metric for an experiment."""
    exp = session.query(type(session).query(session).filter(type(session).query(session)).first().__class__.__bases__[0])  # noqa: simplification
    # Use simpler approach
    from app.database.models import Experiment
    exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    from app.database.models import Metric as ModelMetric
    m = ModelMetric(experiment_id=experiment_id, step=metric_data.step, key=metric_data.key, value=metric_data.value)
    session.add(m)
    session.commit()

    return {"status": "ok", "metric_id": m.id}


@app.post("/api/v1/experiments/{experiment_id}/run")
def run_experiment_pipeline(experiment_id: str, session: Session = Depends(_get_session)):
    """Run the full pipeline for an existing experiment."""
    from app.database.models import Experiment
    exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    config = json.loads(exp.config_json) if exp.config_json else {}
    config["model_name"] = config.get("model_name", exp.name)

    ctx = run_pipeline(
        algorithm=exp.algorithm,
        config=config,
        session=session,
    )

    return {
        "experiment_id": ctx.experiment_id,
        "model_uri": ctx.model_uri,
        "final_metrics": ctx.final_metrics,
    }


# ─── Model Endpoints ─────────────────────────────────────────────────────────


@app.post("/api/v1/models", status_code=201)
def register_model(register_data: ModelRegister, session: Session = Depends(_get_session)):
    """Register a model version in the registry."""
    from app.database.models import ModelVersion

    # Get next version number
    existing = (
        session.query(ModelVersion)
        .filter(ModelVersion.model_name == register_data.model_name)
        .order_by(ModelVersion.version.desc())
        .first()
    )
    next_version = (existing.version if existing else 0) + 1

    model_version = ModelVersion(
        experiment_id=str(uuid.uuid4()),  # Placeholder — link to experiment separately
        model_name=register_data.model_name,
        version=next_version,
        algorithm=AlgorithmType.SKLEARN,
        model_uri=register_data.model_uri,
        tags_json=json.dumps(register_data.tags),
        registered_at=datetime.utcnow(),
    )
    session.add(model_version)
    session.commit()

    return {
        "id": model_version.id,
        "model_name": model_version.model_name,
        "version": model_version.version,
        "model_uri": model_version.model_uri,
        "tags": register_data.tags,
    }


@app.get("/api/v1/models")
def list_models(model_name: Optional[str] = None, session: Session = Depends(_get_session)):
    """List registered models."""
    versions = ModelVersionService.list_all(session, model_name=model_name)
    return [
        {
            "id": v.id,
            "model_name": v.model_name,
            "version": v.version,
            "algorithm": v.algorithm.value,
            "metrics_json": json.loads(v.metrics_json) if v.metrics_json else None,
            "model_uri": v.model_uri,
            "registered_at": v.registered_at.isoformat() if v.registered_at else None,
        }
        for v in versions
    ]


# ─── Deployment Endpoints ─────────────────────────────────────────────────────


@app.post("/api/v1/models/{version_id}/deploy")
def deploy_model(version_id: str, deploy_data: ModelDeploy, session: Session = Depends(_get_session)):
    """Deploy a model version."""
    from app.database.models import ModelVersion, Deployment

    mv = session.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail=f"Model version {version_id} not found")

    deployment = Deployment(
        model_version_id=version_id,
        environment=deploy_data.environment,
        status=DeploymentStatus.DEPLOYED,
        deployed_by=deploy_data.deployed_by,
        notes=deploy_data.notes,
    )
    session.add(deployment)
    session.commit()

    return {
        "id": deployment.id,
        "model_version_id": version_id,
        "environment": deploy_data.environment,
        "status": "deployed",
    }


@app.get("/api/v1/deployments")
def list_deployments(environment: Optional[str] = None, session: Session = Depends(_get_session)):
    """List deployments."""
    deps = DeploymentService.list_all(session, environment=environment)
    return [
        {
            "id": d.id,
            "model_version_id": d.model_version_id,
            "environment": d.environment,
            "status": d.status.value,
            "deployed_at": d.deployed_at.isoformat() if d.deployed_at else None,
        }
        for d in deps
    ]


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
