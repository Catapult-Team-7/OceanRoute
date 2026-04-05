from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from db.database import get_repo
from db.demo_data import DemoOceanRepository
from ingest.real_training_data import RealDataLoadError
from ml.trainer import MLTrainerService
from pipeline.artifacts import data_manifest_path, load_published_map_bundle, read_json, training_manifest_path
from pipeline.jobs import prepare_training_artifacts, publish_verified_map

router = APIRouter()


class TrainingConfig(BaseModel):
    epochs: int = Field(default=10, ge=5, le=200)
    learning_rate: float = Field(default=0.005, gt=0.0001, le=1.0)
    month_window: int = Field(default=12, ge=3, le=24)
    resolution: str = Field(default="2deg")
    quick_test: bool = False


class ApiSourceUpdate(BaseModel):
    id: str | None = None
    name: str | None = None
    enabled: bool = False
    status: str | None = None
    url: str | None = None
    notes: str | None = None


class PublishConfig(BaseModel):
    date: str | None = None
    resolution: str = Field(default="2deg")
    region: str = Field(default="global")


class PrepareConfig(BaseModel):
    epochs: int = Field(default=10, ge=5, le=400)
    learning_rate: float = Field(default=0.005, gt=0.00001, le=1.0)
    month_window: int = Field(default=12, ge=3, le=24)
    resolution: str = Field(default="2deg")
    quick_test: bool = False


def get_trainer(repo: Annotated[DemoOceanRepository, Depends(get_repo)]):
    return repo.trainer


@router.get("/ml/status")
async def ml_status(trainer: Annotated[MLTrainerService, Depends(get_trainer)]):
    return trainer.get_status()


@router.get("/ml/apis")
async def ml_apis(trainer: Annotated[MLTrainerService, Depends(get_trainer)]):
    return {"apis": trainer.list_required_apis()}


@router.put("/ml/apis")
async def ml_update_apis(updates: list[ApiSourceUpdate], trainer: Annotated[MLTrainerService, Depends(get_trainer)]):
    return {"apis": trainer.update_api_configs([item.model_dump(exclude_none=True) for item in updates])}


@router.post("/ml/train")
async def ml_train(config: TrainingConfig, trainer: Annotated[MLTrainerService, Depends(get_trainer)]):
    state, started = trainer.start_training(config.model_dump())
    return {"started": started, "training": state}


@router.post("/ml/prepare")
async def ml_prepare(config: PrepareConfig):
    return await run_in_threadpool(prepare_training_artifacts, config.model_dump())


@router.post("/ml/publish")
async def ml_publish(config: PublishConfig):
    try:
        return await run_in_threadpool(
            publish_verified_map,
            date=config.date,
            resolution=config.resolution,
            region=config.region,
        )
    except RealDataLoadError as exc:
        return {
            "status": "blocked",
            "message": str(exc),
            "metadata": {
                "verified_map": False,
                "map_source": "unpublished_real_grid",
                "source_summary": f"Verified map publish is blocked: {exc}",
            },
        }


@router.get("/ml/artifacts")
async def ml_artifacts():
    published_global_1deg = load_published_map_bundle(region="global", resolution="1deg")
    published_global_2deg = load_published_map_bundle(region="global", resolution="2deg")
    return {
        "data_manifest": read_json(data_manifest_path()),
        "training_manifest": read_json(training_manifest_path()),
        "published_global_1deg": published_global_1deg.get("metadata") if published_global_1deg else None,
        "published_global_2deg": published_global_2deg.get("metadata") if published_global_2deg else None,
    }
