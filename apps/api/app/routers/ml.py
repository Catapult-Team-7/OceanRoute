from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import (
    DatasetArtifact,
    DatasetBuildRequest,
    DatasetExportRequest,
    ModelEvaluateResponse,
    ModelExportResponse,
    ModelPromotionResponse,
    ModelRegistryEntry,
    ModelTrainRequest,
    ModelTrainResponse,
)
from app.services.dataset_service import build_dataset, inspect_dataset, list_datasets
from app.services.model_registry_service import (
    evaluate_model,
    export_model_artifact,
    list_models,
    promote_model,
    train_model,
)

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post("/datasets/build", response_model=DatasetArtifact)
def build_dataset_endpoint(request: DatasetBuildRequest, db: Session = Depends(get_db)) -> DatasetArtifact:
    try:
        return build_dataset(request, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/datasets/export", response_model=DatasetArtifact)
def export_dataset_endpoint(request: DatasetExportRequest, db: Session = Depends(get_db)) -> DatasetArtifact:
    try:
        return build_dataset(request, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets", response_model=list[DatasetArtifact])
def list_datasets_endpoint(region_id: str | None = Query(default=None), db: Session = Depends(get_db)) -> list[DatasetArtifact]:
    return list_datasets(db, region_id=region_id)


@router.get("/datasets/{dataset_id}")
def inspect_dataset_endpoint(dataset_id: str, db: Session = Depends(get_db)) -> dict[str, object]:
    try:
        return inspect_dataset(dataset_id, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/train", response_model=ModelTrainResponse)
def train_model_endpoint(request: ModelTrainRequest, db: Session = Depends(get_db)) -> ModelTrainResponse:
    try:
        return train_model(request, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models", response_model=list[ModelRegistryEntry])
def list_models_endpoint(region_id: str | None = Query(default=None), db: Session = Depends(get_db)) -> list[ModelRegistryEntry]:
    return list_models(db, region_id=region_id)


@router.get("/models/{model_id}/evaluate", response_model=ModelEvaluateResponse)
def evaluate_model_endpoint(model_id: str, db: Session = Depends(get_db)) -> ModelEvaluateResponse:
    try:
        return evaluate_model(model_id, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/models/{model_id}/export", response_model=ModelExportResponse)
def export_model_endpoint(model_id: str, db: Session = Depends(get_db)) -> ModelExportResponse:
    try:
        return export_model_artifact(model_id, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/models/{model_id}/promote", response_model=ModelPromotionResponse)
def promote_model_endpoint(model_id: str, db: Session = Depends(get_db)) -> ModelPromotionResponse:
    try:
        return promote_model(model_id, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
