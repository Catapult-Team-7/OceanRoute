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
        if row.predicted_flux is not None:
            ml_scores = {
                "predicted_flux": row.predicted_flux,
                "observed_flux": row.observed_flux if row.observed_flux is not None else row.predicted_flux,
                "weakening_score": row.weakening_score or 0.0,
                "route_priority": row.route_priority or 0.0,
            }
        else:
            ml_scores = repo.trainer.score_row(row)
        features.append(
            FluxFeature(
                geometry=FeatureGeometry(coordinates=(row.lon, row.lat)),
                properties=FeatureProperties(
                    flux=ml_scores["predicted_flux"],
                    observed_flux=ml_scores["observed_flux"],
                    predicted_flux=ml_scores["predicted_flux"],
                    sst=row.sst,
                    current_u=getattr(row, "current_u", None),
                    current_v=getattr(row, "current_v", None),
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
            date=repo.last_grid_metadata.get("date", date or request.app.state.repo.now.strftime("%Y-%m")),
            units="mol CO2/m²/yr",
            mean_flux=round(mean_flux, 3),
            sink_area_pct=round(sink_area_pct, 1),
            inference_mode=repo.last_grid_metadata.get(
                "inference_mode", repo.trainer.state.model_summary.get("mode", "real_grid_pending")
            ),
            trained_model_ready=repo.last_grid_metadata.get("trained_model_ready", repo.trainer.state.model_ready),
            verified_map=repo.last_grid_metadata.get("verified_map", False),
            map_source=repo.last_grid_metadata.get("map_source", "real_grid_unavailable"),
            source_summary=repo.last_grid_metadata.get(
                "source_summary",
                "No verified CO2 ocean layer is available yet. The map remains empty until a checkpoint-backed real grid is ready.",
            ),
        ),
        features=features,
    )
