from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import (
    Base,
    BaselineArtifactModel,
    DatasetArtifactModel,
    FeedbackEventModel,
    ForecastRunModel,
    ForecastStepModel,
    MissionOutcomeModel,
    ModelRegistryModel,
    ObservationModel,
    PredictionArtifactModel,
    RoutePlanModel,
    TrainingRunModel,
)


def _engine_kwargs() -> dict[str, object]:
    if settings.database_url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {}


engine = create_engine(settings.database_url, future=True, **_engine_kwargs())
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    with engine.connect() as connection:
        inspector = inspect(connection)
        existing_tables = set(inspector.get_table_names())
    required_tables = {
        ForecastRunModel.__tablename__,
        ForecastStepModel.__tablename__,
        RoutePlanModel.__tablename__,
        ObservationModel.__tablename__,
        MissionOutcomeModel.__tablename__,
        FeedbackEventModel.__tablename__,
        BaselineArtifactModel.__tablename__,
        DatasetArtifactModel.__tablename__,
        ModelRegistryModel.__tablename__,
        PredictionArtifactModel.__tablename__,
        TrainingRunModel.__tablename__,
    }
    missing_tables = sorted(required_tables - existing_tables)
    if missing_tables:
        joined = ", ".join(missing_tables)
        raise RuntimeError(
            f"Database schema is not initialized. Missing tables: {joined}. "
            "Run `python -m alembic upgrade head` or `powershell -ExecutionPolicy Bypass -File scripts\\bootstrap-db.ps1`."
        )


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
