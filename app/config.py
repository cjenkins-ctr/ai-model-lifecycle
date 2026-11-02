"""Configuration management for the AI Model Lifecycle project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dot_env = lambda: load_dotenv(_env_path)
    load_dot_env()


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///models.db"))


@dataclass(frozen=True)
class LoggingConfig:
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


@dataclass(frozen=True)
class AwsConfig:
    region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    s3_bucket: str | None = field(default_factory=lambda: os.getenv("MODEL_S3_BUCKET") or None)


@dataclass(frozen=True)
class RegistryConfig:
    path: str = field(default_factory=lambda: os.getenv("MODEL_REGISTRY_PATH", "./registry"))


@dataclass(frozen=True)
class AppSettings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    aws: AwsConfig = field(default_factory=AwsConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    wandb_api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))


def get_settings() -> AppSettings:
    """Return a singleton AppSettings instance."""
    return AppSettings()
