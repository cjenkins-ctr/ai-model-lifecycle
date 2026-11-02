"""Database session and engine management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def get_engine():
    """Create and return a SQLAlchemy engine."""
    settings = get_settings()
    engine = create_engine(settings.db.url, echo=False)
    return engine


def get_session_factory():
    """Create and return a session factory."""
    engine = get_engine()
    return sessionmaker(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
