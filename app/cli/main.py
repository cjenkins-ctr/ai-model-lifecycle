"""CLI tooling for the AI Model Lifecycle project."""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from app.config import get_settings
from app.database.models import AlgorithmType, ExperimentStatus
from app.database.session import get_session
from app.logging_config import setup_logging
from app.pipeline.orchestrator import AlgorithmType as AlgoType, run_pipeline
from app.services.query_service import ExperimentService, ModelVersionService, DeploymentService

console = Console()


def _init_db():
    """Ensure the database tables exist."""
    from app.database.models import init_db
    settings = get_settings()
    init_db(settings.db.url)


# ─── CLI Group ────────────────────────────────────────────────────────────────


@click.group()
@click.version_option("0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """AI Model Lifecycle — Build, Train & Deploy Models."""
    setup_logging()
    if verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        from app.config import get_settings as _gs
        _gs.__dict__  # force reload
        setup_logging()


# ─── Train Command ────────────────────────────────────────────────────────────


@main.command()
@click.option("--algorithm", "-a", "algorithm",
              type=click.Choice([a.value for a in AlgorithmType]),
              default="sklearn", help="Algorithm family to use.")
@click.option("--dataset", "-d", "dataset",
              type=click.Choice(["iris", "wine", "digits", "breast_cancer", "moon", "circles"]),
              default="iris", help="Dataset to train on.")
@click.option("--model-type", "-m", "model_type",
              type=click.Choice(["logistic_regression", "random_forest", "knn", "svm"]),
              default="logistic_regression", help="Specific model type (for sklearn).")
@click.option("--epochs", "-e", "epochs", default=1, help="Training epochs (sklearn ignores).")
@click.option("--deploy-env", "--env", "deploy_env",
              type=click.Choice(["staging", "production", "canary"]),
              default=None, help="Deploy to environment after training.")
@click.option("--registry-path", "-r", "registry_path", default="./registry",
              help="Path to model registry.")
def train_cmd(algorithm: str, dataset: str, model_type: str, epochs: int,
              deploy_env: str | None, registry_path: str):
    """Run a training pipeline end-to-end."""
    _init_db()

    settings = get_settings()
    algo = AlgorithmType(algorithm)

    config = {
        "dataset": dataset,
        "model_type": model_type,
        "epochs": epochs,
        "model_name": f"{algorithm}-{model_type}",
    }

    console.print(f"[bold blue]Starting training pipeline[/bold blue]")
    console.print(f"  Algorithm: {algorithm}")
    console.print(f"  Dataset:   {dataset}")
    console.print(f"  Model:     {model_type}")
    console.print(f"  Epochs:    {epochs}")

    try:
        with get_session() as session:
            ctx = run_pipeline(
                algorithm=algo,
                config=config,
                session=session,
                registry_path=registry_path,
                deploy_env=deploy_env,
            )
        console.print(f"\n[bold green]✓ Pipeline completed[/bold green]")
        console.print(f"  Experiment ID: {ctx.experiment_id}")
        console.print(f"  Model URI:     {ctx.model_uri}")
        console.print(f"  Final metrics: {json.dumps(ctx.final_metrics, indent=2)}")
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed: {e}[/bold red]")
        sys.exit(1)


# ─── Experiments Command ──────────────────────────────────────────────────────


@main.group()
def experiments():
    """Query experiment data."""
    pass


@experiments.command("list")
@click.option("--status", "-s", "status",
              type=click.Choice([s.value for s in ExperimentStatus]),
              default=None, help="Filter by status.")
def exp_list(status: str | None):
    """List all experiments."""
    _init_db()

    with get_session() as session:
        filter_status = ExperimentStatus(status) if status else None
        exps = ExperimentService.get_all(session, status=filter_status)
        rows = [
            {
                "id": exp.id,
                "name": exp.name,
                "algorithm": exp.algorithm.value,
                "status": exp.status.value,
                "created_at": exp.created_at.strftime("%Y-%m-%d %H:%M") if exp.created_at else "—",
            }
            for exp in exps
        ]

    if not rows:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Algorithm", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Created", style="dim")

    for row in rows:
        table.add_row(
            row["id"][:8],
            row["name"],
            row["algorithm"],
            row["status"],
            row["created_at"],
        )

    console.print(table)
    console.print(f"\nTotal: {len(rows)} experiment(s)")


@experiments.command("metrics")
@click.argument("experiment_id")
def exp_metrics(experiment_id: str):
    """Show metrics for an experiment."""
    _init_db()

    from app.services.query_service import ExperimentService

    with get_session() as session:
        metrics = ExperimentService.get_metrics(session, experiment_id)

    if not metrics:
        console.print(f"[yellow]No metrics found for experiment {experiment_id}[/yellow]")
        return

    table = Table(title=f"Metrics — {experiment_id[:8]}")
    table.add_column("Step", style="dim")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    for m in metrics:
        table.add_row(str(m.step), m.key, f"{m.value:.6f}")

    console.print(table)


# ─── Models Command ───────────────────────────────────────────────────────────


@main.group()
def models():
    """Query registered models."""
    pass


@models.command("list")
@click.option("--name", "-n", "model_name", default=None, help="Filter by model name.")
def mdl_list(model_name: str | None):
    """List registered model versions."""
    _init_db()

    with get_session() as session:
        versions = ModelVersionService.list_all(session, model_name=model_name)
        rows = [
            {
                "model_name": v.model_name,
                "version": v.version,
                "algorithm": v.algorithm.value,
                "metrics": "; ".join(f"{k}={val:.4f}" for k, val in json.loads(v.metrics_json).items()) if v.metrics_json else "—",
                "uri": v.model_uri[:60],
            }
            for v in versions
        ]

    if not rows:
        console.print("[yellow]No models registered.[/yellow]")
        return

    table = Table(title="Registered Models")
    table.add_column("Name", style="cyan")
    table.add_column("Version", justify="right")
    table.add_column("Algorithm", style="magenta")
    table.add_column("Metrics", style="green")
    table.add_column("URI", style="dim")

    for row in rows:
        table.add_row(
            row["model_name"],
            str(row["version"]),
            row["algorithm"],
            row["metrics"],
            row["uri"],
        )

    console.print(table)


# ─── Deployments Command ──────────────────────────────────────────────────────


@main.group()
def deployments():
    """Query deployments."""
    pass


@deployments.command("list")
@click.option("--env", "-e", "environment", default=None, help="Filter by environment.")
def dep_list(environment: str | None):
    """List deployments."""
    _init_db()

    with get_session() as session:
        deps = DeploymentService.list_all(session, environment=environment)
        rows = [
            {
                "id": d.id,
                "env": d.environment,
                "status": d.status.value,
                "deployed_at": d.deployed_at.strftime("%Y-%m-%d %H:%M") if d.deployed_at else "—",
            }
            for d in deps
        ]

    if not rows:
        console.print("[yellow]No deployments found.[/yellow]")
        return

    table = Table(title="Deployments")
    table.add_column("ID", style="cyan")
    table.add_column("Environment", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Deployed At", style="dim")

    for row in rows:
        table.add_row(
            row["id"][:8],
            row["env"],
            row["status"],
            row["deployed_at"],
        )

    console.print(table)


# ─── Server Command ───────────────────────────────────────────────────────────


@main.command()
@click.option("--host", "-h", default="0.0.0.0", help="Bind host.")
@click.option("--port", "-p", default=8000, help="Bind port.")
def serve(host: str, port: int):
    """Start the FastAPI model registry server."""
    import uvicorn
    _init_db()

    console.print(f"[bold blue]Starting API server on {host}:{port}[/bold blue]")
    uvicorn.run(
        "app.serve:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
