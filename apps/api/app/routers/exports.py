from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.db import get_db
from app.services.export_service import build_pdf_brief_bytes, forecast_to_geojson
from app.services.forecast_service import latest_forecast

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/geojson")
def export_geojson_endpoint(
    horizon_hour: int | None = Query(default=None, ge=1, le=72),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    snapshot = latest_forecast(db, horizon_hour=horizon_hour, min_confidence=min_confidence)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No forecast has been generated yet.")
    return forecast_to_geojson(snapshot)


@router.get("/pdf-brief")
def export_pdf_brief_endpoint(db: Session = Depends(get_db)) -> Response:
    snapshot = latest_forecast(db)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No forecast has been generated yet.")
    pdf_bytes = build_pdf_brief_bytes(snapshot)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=mission-brief.pdf"},
    )
