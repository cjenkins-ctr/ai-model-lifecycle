"""Tests for the AI Model Lifecycle project."""

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database.models import (
    AlgorithmType,
    Artifact,
    Deployment,
    DeploymentStatus,
    Experiment,
    ExperimentStatus,
    Metric,
    ModelVersion,
    Base,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def session(engine: any) -> Session:
    """Create a test database session."""
    factory = sessionmaker(bind=engine)
    sess = factory()
    yield sess
    sess.close()


class TestExperimentModel:
    """Tests for Experiment model CRUD."""

    def test_create_experiment(self, session: Session):
        exp = Experiment(
            id=str(uuid.uuid4()),
            name="test-experiment",
            algorithm=AlgorithmType.SKLEARN,
        )
        session.add(exp)
        session.commit()

        found = session.query(Experiment).filter(Experiment.name == "test-experiment").first()
        assert found is not None
        assert found.algorithm == AlgorithmType.SKLEARN

    def test_experiment_status_transitions(self, session: Session):
        exp = Experiment(
            id=str(uuid.uuid4()),
            name="status-test",
            algorithm=AlgorithmType.SKLEARN,
        )
        session.add(exp)
        session.commit()

        exp.status = ExperimentStatus.RUNNING
        session.commit()
        assert exp.status == ExperimentStatus.RUNNING

        exp.status = ExperimentStatus.COMPLETED
        session.commit()
        assert exp.status == ExperimentStatus.COMPLETED


class TestMetricModel:
    """Tests for Metric model."""

    def test_log_metric(self, session: Session):
        exp = Experiment(
            id=str(uuid.uuid4()),
            name="metric-test",
            algorithm=AlgorithmType.SKLEARN,
        )
        session.add(exp)
        session.commit()

        metric = Metric(
            experiment_id=exp.id,
            step=1,
            key="train/loss",
            value=0.5,
        )
        session.add(metric)
        session.commit()

        metrics = session.query(Metric).filter(Metric.experiment_id == exp.id).all()
        assert len(metrics) == 1
        assert metrics[0].value == 0.5


class TestModelVersion:
    """Tests for ModelVersion model."""

    def test_register_model_version(self, session: Session):
        exp = Experiment(
            id=str(uuid.uuid4()),
            name="model-test",
            algorithm=AlgorithmType.SKLEARN,
        )
        session.add(exp)
        session.commit()

        mv = ModelVersion(
            experiment_id=exp.id,
            model_name="test-model",
            version=1,
            algorithm=AlgorithmType.SKLEARN,
            model_uri="/path/to/model.pkl",
        )
        session.add(mv)
        session.commit()

        found = session.query(ModelVersion).filter(
            ModelVersion.model_name == "test-model",
            ModelVersion.version == 1,
        ).first()
        assert found is not None
        assert found.model_uri == "/path/to/model.pkl"


class TestPipelineOrchestrator:
    """Tests for the pipeline orchestrator."""

    def test_run_sklearn_pipeline(self, session: Session, tmp_path: any):
        """Run a complete sklearn training pipeline."""
        from app.pipeline.orchestrator import run_pipeline

        config = {
            "dataset": "iris",
            "model_type": "logistic_regression",
            "epochs": 3,
            "model_name": "test-sklearn",
        }

        ctx = run_pipeline(
            algorithm=AlgorithmType.SKLEARN,
            config=config,
            session=session,
            registry_path=str(tmp_path / "registry"),
            deploy_env=None,
        )

        # Verify experiment was created
        exp = session.query(Experiment).get(ctx.experiment_id)
        assert exp is not None
        assert exp.status == ExperimentStatus.COMPLETED

        # Verify metrics were logged
        metrics = session.query(Metric).filter(Metric.experiment_id == ctx.experiment_id).all()
        assert len(metrics) > 0

        # Verify model version was registered
        from app.database.models import ModelVersion
        mv = session.query(ModelVersion).filter(
            ModelVersion.experiment_id == ctx.experiment_id
        ).first()
        assert mv is not None
        assert mv.model_name == "test-sklearn"


class TestServiceQueries:
    """Tests for the query service."""

    def test_list_experiments(self, session: Session):
        """Test experiment listing."""
        for i in range(3):
            exp = Experiment(
                id=str(uuid.uuid4()),
                name=f"exp-{i}",
                algorithm=AlgorithmType.SKLEARN,
            )
            session.add(exp)
        session.commit()

        from app.services.query_service import ExperimentService
        exps = ExperimentService.get_all(session)
        assert len(exps) == 3

    def test_get_latest_model(self, session: Session):
        """Test getting the latest model version."""
        from app.services.query_service import ModelVersionService

        exp1 = Experiment(
            id=str(uuid.uuid4()),
            name="model-exp-1",
            algorithm=AlgorithmType.SKLEARN,
        )
        session.add(exp1)
        session.commit()

        mv1 = ModelVersion(
            experiment_id=exp1.id,
            model_name="classifier",
            version=1,
            algorithm=AlgorithmType.SKLEARN,
            model_uri="/path/v1",
        )
        session.add(mv1)
        session.commit()

        exp2 = Experiment(
            id=str(uuid.uuid4()),
            name="model-exp-2",
            algorithm=AlgorithmType.SKLEARN,
        )
        session.add(exp2)
        session.commit()

        mv2 = ModelVersion(
            experiment_id=exp2.id,
            model_name="classifier",
            version=2,
            algorithm=AlgorithmType.SKLEARN,
            model_uri="/path/v2",
        )
        session.add(mv2)
        session.commit()

        latest = ModelVersionService.get_latest(session, "classifier")
        assert latest is not None
        assert latest.version == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
