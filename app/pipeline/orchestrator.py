"""Pipeline orchestration: build → train → evaluate → register → deploy."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.database.models import (
    AlgorithmType,
    Artifact,
    DeploymentStatus,
    Experiment,
    ExperimentStatus,
    Metric,
    ModelVersion,
    Deployment,
)
from app.logging_config import get_logger

logger = get_logger(__name__)


class PipelineContext:
    """Shared context passed through pipeline stages."""

    def __init__(
        self,
        experiment_id: str,
        algorithm: AlgorithmType,
        config: dict,
        session: Session,
        registry_path: str = "./registry",
    ):
        self.experiment_id = experiment_id
        self.algorithm = algorithm
        self.config = config
        self.session = session
        self.registry_path = Path(registry_path)
        self.checkpoint_path: Path | None = None
        self.metrics: dict[str, list[tuple[int, float]]] = {}
        self.final_metrics: dict[str, float] = {}
        self.model_uri: str = ""
        self.model_name: str = config.get("model_name", "baseline")
        self.tags: list[str] = config.get("tags", [])

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """Log a metric within the experiment."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append((step, value))

        metric = Metric(
            experiment_id=self.experiment_id,
            step=step,
            key=key,
            value=value,
        )
        self.session.add(metric)

    def save_checkpoint(self, checkpoint_file: str, artifact_type: str = "checkpoint") -> None:
        """Register a checkpoint file as an artifact."""
        artifact = Artifact(
            experiment_id=self.experiment_id,
            name=Path(checkpoint_file).name,
            file_path=str(checkpoint_file),
            artifact_type=artifact_type,
        )
        self.session.add(artifact)

    def commit(self) -> None:
        """Commit pending session changes."""
        self.session.commit()


# ─── Pipeline Stages ──────────────────────────────────────────────────────────


def stage_build(ctx: PipelineContext) -> PipelineContext:
    """Stage 1: Build — prepare experiment record and data pipeline."""
    logger.info("stage_build", experiment_id=ctx.experiment_id, algorithm=ctx.algorithm.value)

    experiment = Experiment(
        id=ctx.experiment_id,
        name=f"{ctx.model_name}-{ctx.algorithm.value}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        algorithm=ctx.algorithm,
        status=ExperimentStatus.RUNNING,
        config_json=json.dumps(ctx.config),
    )
    ctx.session.add(experiment)
    ctx.session.commit()

    # Create registry directory
    ctx.registry_path.mkdir(parents=True, exist_ok=True)
    logger.info("experiment_created", experiment_id=ctx.experiment_id)

    return ctx


def stage_train(ctx: PipelineContext) -> PipelineContext:
    """Stage 2: Train — execute the training loop.

    Uses the registered algorithm from app.models to run training.
    Metrics are logged through ctx.log_metric().
    """
    logger.info("stage_train", experiment_id=ctx.experiment_id)

    try:
        from app.models.registry import get_trainer

        trainer = get_trainer(ctx.algorithm)
        trainer.train(ctx)

        ctx.session.commit()
        logger.info("training_completed", experiment_id=ctx.experiment_id)
    except Exception as e:
        ctx.session.rollback()
        logger.error("training_failed", experiment_id=ctx.experiment_id, error=str(e))
        experiment = ctx.session.query(Experiment).filter(Experiment.id == ctx.experiment_id).first()
        if experiment:
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = str(e)
            ctx.session.commit()
        raise

    return ctx


def stage_evaluate(ctx: PipelineContext) -> PipelineContext:
    """Stage 3: Evaluate — compute final evaluation metrics."""
    logger.info("stage_evaluate", experiment_id=ctx.experiment_id)

    try:
        from app.models.registry import get_evaluator

        evaluator = get_evaluator(ctx.algorithm)
        eval_results = evaluator.evaluate(ctx)

        # Save final metrics
        for key, value in eval_results.items():
            ctx.log_metric(key, value, step=-1)  # step=-1 for final metrics
            ctx.final_metrics[key] = value
            logger.info("final_metric", key=key, value=value)

        # Mark experiment as completed
        experiment = ctx.session.query(Experiment).get(ctx.experiment_id)
        if experiment:
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            ctx.session.commit()

        logger.info("evaluation_completed", experiment_id=ctx.experiment_id)
    except Exception as e:
        logger.error("evaluation_failed", experiment_id=ctx.experiment_id, error=str(e))
        raise

    return ctx


def stage_register(ctx: PipelineContext) -> PipelineContext:
    """Stage 4: Register — save model to registry and create model version."""
    logger.info("stage_register", experiment_id=ctx.experiment_id)

    # Get next version number
    existing = (
        ctx.session.query(ModelVersion)
        .filter(ModelVersion.model_name == ctx.model_name)
        .order_by(ModelVersion.version.desc())
        .first()
    )
    next_version = (existing.version if existing else 0) + 1

    # Set model URI to checkpoint path
    ctx.model_uri = str(ctx.checkpoint_path) if ctx.checkpoint_path else "artifact://not_set"

    model_version = ModelVersion(
        experiment_id=ctx.experiment_id,
        model_name=ctx.model_name,
        version=next_version,
        algorithm=ctx.algorithm,
        metrics_json=json.dumps(ctx.final_metrics),
        model_uri=ctx.model_uri,
        tags_json=json.dumps(ctx.tags),
        registered_at=datetime.utcnow(),
    )
    ctx.session.add(model_version)
    ctx.session.commit()

    logger.info(
        "model_registered",
        model_name=ctx.model_name,
        version=next_version,
        experiment_id=ctx.experiment_id,
    )

    return ctx


def stage_deploy(
    ctx: PipelineContext,
    environment: str = "staging",
    deployed_by: str = "system",
) -> Deployment:
    """Stage 5: Deploy — deploy the registered model to an environment."""
    logger.info("stage_deploy", experiment_id=ctx.experiment_id, environment=environment)

    deployment = Deployment(
        model_version_id=ctx.experiment_id,  # Points to model version via FK
        environment=environment,
        status=DeploymentStatus.DEPLOYED,
        deployed_by=deployed_by,
        notes=f"Deployed via pipeline for experiment {ctx.experiment_id}",
    )
    ctx.session.add(deployment)
    ctx.session.commit()

    logger.info("model_deployed", deployment_id=deployment.id, environment=environment)

    return deployment


# ─── Pipeline Runner ─────────────────────────────────────────────────────────


def run_pipeline(
    algorithm: AlgorithmType,
    config: dict,
    session: Session,
    registry_path: str = "./registry",
    deploy_env: str | None = None,
) -> PipelineContext:
    """Run the full build → train → evaluate → register → deploy pipeline.

    Stages are sequential and each can be individually disabled by raising
    SkipStage (not implemented — use run_pipeline_stages for partial runs).
    """
    ctx = PipelineContext(
        experiment_id=str(uuid.uuid4()),
        algorithm=algorithm,
        config=config,
        session=session,
        registry_path=registry_path,
    )

    # Stage 1: Build
    ctx = stage_build(ctx)

    # Stage 2: Train
    ctx = stage_train(ctx)

    # Stage 3: Evaluate
    ctx = stage_evaluate(ctx)

    # Stage 4: Register
    ctx = stage_register(ctx)

    # Stage 5: Deploy (optional)
    if deploy_env:
        stage_deploy(ctx, environment=deploy_env)

    return ctx
