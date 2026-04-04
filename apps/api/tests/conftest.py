from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient


TEST_DB_PATH = Path(__file__).resolve().parent / "test_oceanroute.db"
TEST_DATA_ROOT = Path(__file__).resolve().parent / "test-data"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

os.environ["OCEANROUTE_DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["OCEANROUTE_INGEST_MODE"] = "sample"
os.environ["OCEANROUTE_SCHEDULER_ENABLED"] = "false"
os.environ["OCEANROUTE_WRITE_DEBUG_ARTIFACTS"] = "false"
os.environ["OCEANROUTE_DATA_ROOT"] = str(TEST_DATA_ROOT)

from app.db import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base  # noqa: E402


def _upgrade_test_db() -> None:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", os.environ["OCEANROUTE_DATABASE_URL"])
    command.upgrade(config, "head")


@pytest.fixture(autouse=True)
def reset_db() -> None:
    engine.dispose()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    _upgrade_test_db()
    yield
    engine.dispose()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
