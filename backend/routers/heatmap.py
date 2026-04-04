from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from db.crud import get_flux_grid
from db.database import get_repo
from db.demo_data import DemoOceanRepository
from db.models import FeatureGeometry, FeatureProperties, FluxFeature, HeatmapMetadata, HeatmapResponse

router = APIRouter()


@router.get("/heatmap", response_model=HeatmapResponse)
async def heatmap(
    request: Request,
    date: str | None = Query(default=None, description="YYYY-MM"),
    resolution: str = Query(default="1deg", pattern="^(0.25deg|0.5deg|1deg|2deg)$"),
    region: str = Query(default="global", pattern="^(global|pacific|atlantic|indian)$"),
    repo: Annotated[DemoOceanRepository, Depends(get_repo)] = None,
):
    rows = await get_flux_grid(repo, date=date, resolution=resolution, region=region)
    features = []
    for row in rows:
        ml_scores = repo.trainer.score_row(row)
        features.append(
            FluxFeature(
                geometry=FeatureGeometry(coordinates=(row.lon, row.lat)),
                properties=FeatureProperties(
                    flux=ml_scores["predicted_flux"],
                    observed_flux=ml_scores["observed_flux"],
                    predicted_flux=ml_scores["predicted_flux"],
                    sst=row.sst,
                    anomaly_score=row.anomaly_score,
                    weakening_score=ml_scores["weakening_score"],
                    route_priority=ml_scores["route_priority"],
                    is_anomaly=row.anomaly_score >= 0.5,
                ),
            )
        )
    mean_flux = sum(feature.properties.flux for feature in features) / max(len(features), 1)
    sink_area_pct = sum(1 for feature in features if feature.properties.flux < 0) / max(len(features), 1) * 100
    return HeatmapResponse(
        metadata=HeatmapMetadata(
            date=date or request.app.state.repo.now.strftime("%Y-%m"),
            units="mol CO2/m²/yr",
            mean_flux=round(mean_flux, 3),
            sink_area_pct=round(sink_area_pct, 1),
            inference_mode=repo.trainer.state.model_summary.get("mode", "demo_regression"),
            trained_model_ready=repo.trainer.state.model_ready,
        ),
        features=features,
    )
