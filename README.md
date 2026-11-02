# AI Model Lifecycle

End-to-end AI model development: **build, train, and deploy** with experiment tracking and model registry.

## Architecture

```
app/
├── api/            # FastAPI endpoints (model registry, experiment queries)
├── cli/            # CLI tooling (train, serve, register commands)
├── database/       # SQLAlchemy models, migrations, session management
├── pipeline/       # Build → Train → Evaluate → Register → Deploy stages
├── models/         # Model definition files (placeholder architectures)
├── config/         # Configuration management
├── services/       # Business logic (training service, registry service)
├── utils/          # Logging, helpers, validators
└── __init__.py
```

## Database

SQLite with SQLAlchemy ORM. Schema tracks:

| Table            | Purpose                              |
|------------------|--------------------------------------|
| `experiments`    | Experiment metadata, config, status  |
| `metrics`        | Per-step training/evaluation metrics |
| `model_versions` | Registered model versions & metadata |
| `artifacts`      | Checkpoint files and artifact paths  |
| `deployments`    | Model deployment history             |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run the API server
model-serve

# 4. Run a training job (uses scikit-learn baseline by default)
model-train --algorithm sklearn --dataset iris

# 5. Query experiments via CLI
ai-model-lifecycle experiments list
ai-model-lifecycle models list
```

## API Endpoints

| Method   | Path                    | Description                  |
|----------|-------------------------|------------------------------|
| `POST`   | `/api/v1/experiments`   | Create a new experiment      |
| `GET`    | `/api/v1/experiments`   | List all experiments         |
| `GET`    | `/api/v1/experiments/{id}` | Get experiment details    |
| `POST`   | `/api/v1/experiments/{id}/metrics` | Log metrics        |
| `POST`   | `/api/v1/models`        | Register a model version     |
| `GET`    | `/api/v1/models`        | List registered models       |
| `POST`   | `/api/v1/models/{id}/deploy` | Deploy a model version  |
| `GET`    | `/api/v1/deployments`   | List deployments             |

## Training Pipeline

The pipeline follows a 5-stage process:

1. **Build** — Prepare data, define model architecture
2. **Train** — Execute training loop with metric logging
3. **Evaluate** — Compute evaluation metrics on validation set
4. **Register** — Save model to registry with metadata
5. **Deploy** — Deploy to production or staging environment

## Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Key variables:

| Variable              | Default                  | Description                      |
|-----------------------|--------------------------|----------------------------------|
| `DATABASE_URL`        | `sqlite:///models.db`    | Database connection string       |
| `LOG_LEVEL`           | `INFO`                   | Logging verbosity                |
| `WAND_API_KEY`        | (empty)                  | W&B API key for experiment sync  |
| `AWS_REGION`          | `us-east-1`              | AWS region for S3 artifact store |
| `MODEL_REGISTRY_PATH` | `./registry`             | Local path for model artifacts   |

## Extending with Custom Models

Add a new model class to `app/models/` and register it in the registry:

```python
from app.models import register_model

@register_model("my_model")
class MyModel:
    def build(self, config):
        # Define architecture
        pass

    def train(self, data, config):
        # Training loop
        pass
```

## License

MIT
