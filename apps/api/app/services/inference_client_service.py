from __future__ import annotations

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.schemas import InferencePredictRequest, InferencePredictResponse
from app.services.inference_runtime_service import predict_from_feature_artifact


def predict_with_inference_service(payload: InferencePredictRequest, db: Session) -> InferencePredictResponse:
    if settings.inference_service_url.lower() in {"local", "in-process"}:
        return predict_from_feature_artifact(payload, db)

    response = httpx.post(
        f"{settings.inference_service_url.rstrip('/')}/predict",
        json=payload.model_dump(mode="json"),
        timeout=settings.inference_service_timeout_seconds,
    )
    response.raise_for_status()
    return InferencePredictResponse.model_validate(response.json())
