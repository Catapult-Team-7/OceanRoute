from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db.demo_data import DemoOceanRepository
from db.real_products import build_provisional_real_grid_bundle, build_real_grid_bundle
from ingest.fetch_copernicus import default_copernicus_directory, parse_copernicus_notes
from ml.preprocess import build_monthly_training_tensors

from .artifacts import write_data_manifest, write_published_map_bundle


def _resolve_copernicus_training_directory(notes: str) -> str:
    parsed = parse_copernicus_notes(notes)
    raw = parsed.get("path", "").strip() or str(default_copernicus_directory())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return str(path)


def prepare_training_artifacts(config: dict[str, Any] | None = None) -> dict[str, Any]:
    repo = DemoOceanRepository()
    trainer = repo.trainer
    merged = trainer._merge_training_config(config)
    X, y, data_summary = trainer._build_dataset(
        merged["month_window"],
        merged["resolution"],
        merged.get("quick_test", False),
    )
    era_config = trainer._config_by_id("era5") or {}
    copernicus_config = trainer._config_by_id("copernicus_marine") or {}
    tensor_build = build_monthly_training_tensors(
        socat_url=trainer._config_by_id("socat")["url"],
        noaa_gml_url=trainer._config_by_id("noaa_gml_co2")["url"],
        era_directory=str(era_config.get("notes", "")).strip(),
        copernicus_directory=_resolve_copernicus_training_directory(str(copernicus_config.get("notes", ""))),
        month_window=merged["month_window"],
        resolution=merged["resolution"],
        reference_now=repo.now,
    )
    manifest = {
        "config": merged,
        "dataset": {
            "tabular_samples": int(len(X)),
            "target_samples": int(len(y)),
            **data_summary,
        },
        "tensor_build": {
            "tensor_dir": str(tensor_build.tensor_dir),
            "x_path": str(tensor_build.x_path),
            "y_path": str(tensor_build.y_path),
            "mask_path": str(tensor_build.mask_path),
            "atm_path": str(tensor_build.atm_path),
            "metadata_path": str(tensor_build.metadata_path),
            **tensor_build.summary,
        },
    }
    manifest_path = write_data_manifest(manifest)
    return {
        "status": "prepared",
        "manifest_path": str(manifest_path),
        **manifest,
    }


def publish_verified_map(
    *,
    date: str | None = None,
    resolution: str = "1deg",
    region: str = "global",
) -> dict[str, Any]:
    repo = DemoOceanRepository(now=datetime.now(timezone.utc))
    trainer = repo.trainer
    noaa_config = trainer._config_by_id("noaa_gml_co2") or {}
    socat_config = trainer._config_by_id("socat") or {}
    era_config = trainer._config_by_id("era5") or {}
    copernicus_config = trainer._config_by_id("copernicus_marine") or {}

    try:
        bundle = build_provisional_real_grid_bundle(
            date=date,
            resolution=resolution,
            region=region,
            trainer=trainer,
            noaa_gml_url=str(noaa_config.get("url", "")).strip(),
            era_directory=str(era_config.get("notes", "")).strip(),
            copernicus_directory=_resolve_copernicus_training_directory(str(copernicus_config.get("notes", ""))),
            reference_now=repo.now,
        )
        bundle.metadata["source_summary"] = (
            f"{bundle.metadata.get('source_summary', '')} Published via fast checkpoint-backed real-data mode."
        ).strip()
    except Exception:
        bundle = build_real_grid_bundle(
            date=date,
            resolution=resolution,
            region=region,
            trainer=trainer,
            socat_url=str(socat_config.get("url", "")).strip(),
            noaa_gml_url=str(noaa_config.get("url", "")).strip(),
            era_directory=str(era_config.get("notes", "")).strip(),
            copernicus_directory=_resolve_copernicus_training_directory(str(copernicus_config.get("notes", ""))),
            reference_now=repo.now,
        )
    payload = {
        "metadata": bundle.metadata,
        "rows": [row.model_dump(mode="json") for row in bundle.rows],
        "anomalies": [item.model_dump(mode="json") for item in bundle.anomalies],
        "checkpoint_path": trainer.state.model_summary.get("checkpoint_path"),
    }
    bundle_path = write_published_map_bundle(region=region, resolution=resolution, payload=payload)
    return {
        "status": "published",
        "bundle_path": str(bundle_path),
        "metadata": bundle.metadata,
        "row_count": len(bundle.rows),
        "anomaly_count": len(bundle.anomalies),
    }
