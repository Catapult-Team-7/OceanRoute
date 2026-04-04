from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db, init_db
from app.schemas import InferencePredictRequest, InferencePredictResponse, ModelLoadRequest
from app.services.inference_runtime_service import INFERENCE_SERVICE_VERSION, predict_from_feature_artifact, warm_load_model


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="OceanRoute Inference Service",
    version=INFERENCE_SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "inference", "version": INFERENCE_SERVICE_VERSION}


@app.post("/models/load")
def load_model(payload: ModelLoadRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        return warm_load_model(payload.model_id, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict", response_model=InferencePredictResponse)
def predict(payload: InferencePredictRequest, db: Session = Depends(get_db)) -> InferencePredictResponse:
    try:
        return predict_from_feature_artifact(payload, db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
