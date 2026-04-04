from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from db.database import get_repo
from db.demo_data import DemoOceanRepository
from ml.trainer import MLTrainerService

router = APIRouter()


class TrainingConfig(BaseModel):
    epochs: int = Field(default=18, ge=5, le=200)
    learning_rate: float = Field(default=0.05, gt=0.0001, le=1.0)
    month_window: int = Field(default=12, ge=3, le=24)
    resolution: str = Field(default="2deg")


class ApiSourceUpdate(BaseModel):
    name: str
    enabled: bool = False
    status: str | None = None
    url: str | None = None
    notes: str | None = None


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
