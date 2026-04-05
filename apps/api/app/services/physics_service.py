from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from app.schemas import DEBRIS_CLASS_METADATA, DriftBaselineInput, GridPoint


@dataclass(frozen=True)
class DriftBaselineDiagnostics:
    density: dict[str, float]
    ensemble_spread: dict[str, float]
    beaching_fraction: dict[str, float]
    stokes_drift: dict[str, tuple[float, float]]
    windage_fraction: float
    total_particles: int
    beached_particles: int
    ensemble_members: int


@dataclass(frozen=True)
class PreparedGrid:
    cell_ids: tuple[str, ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    current_u: np.ndarray
    current_v: np.ndarray
    wind_u: np.ndarray
    wind_v: np.ndarray
    shoreline: np.ndarray
    restricted: np.ndarray
    stokes_u: np.ndarray
    stokes_v: np.ndarray
    source_weights: np.ndarray
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def _seed(payload: DriftBaselineInput) -> int:
    return 7000 + payload.horizon_hour + len(payload.grid) + int(payload.source_strength * 1000)


def _windage_fraction(payload: DriftBaselineInput) -> float:
    class_fraction = float(DEBRIS_CLASS_METADATA[payload.debris_class]["windage_factor"])
    return max(0.0, payload.windage_factor + class_fraction)


def _km_to_lat(delta_km: np.ndarray) -> np.ndarray:
    return delta_km / 111.0


def _km_to_lon(delta_km: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    scale = np.clip(np.cos(np.radians(latitude)), 0.2, None)
    return delta_km / (111.0 * scale)


def _source_weights(points: list[GridPoint], payload: DriftBaselineInput) -> np.ndarray:
    current_u = np.asarray([point.current_u for point in points], dtype=np.float32)
    current_v = np.asarray([point.current_v for point in points], dtype=np.float32)
    wind_u = np.asarray([point.wind_u for point in points], dtype=np.float32)
    wind_v = np.asarray([point.wind_v for point in points], dtype=np.float32)
    shoreline = np.asarray([point.shoreline_proximity for point in points], dtype=np.float32)
    restricted = np.asarray([1.0 if point.restricted else 0.0 for point in points], dtype=np.float32)
    current_speed = np.sqrt((current_u**2) + (current_v**2))
    wind_speed = np.sqrt((wind_u**2) + (wind_v**2))
    shoreline_bias = 0.55 + (shoreline * 1.1)
    drift_bias = 0.2 + (current_speed * 0.18) + (wind_speed * 0.015)
    class_bias = 0.12 if payload.debris_class == "high" else 0.05
    restriction_penalty = np.where(restricted > 0.5, 0.82, 1.0)
    weights = np.maximum(0.05, (shoreline_bias + drift_bias + class_bias) * restriction_penalty)
    normalized = weights / np.maximum(np.sum(weights), 1e-6)
    return normalized.astype(np.float32)


def _prepare_grid(payload: DriftBaselineInput) -> PreparedGrid:
    latitudes = np.asarray([point.lat for point in payload.grid], dtype=np.float32)
    longitudes = np.asarray([point.lon for point in payload.grid], dtype=np.float32)
    current_u = np.asarray([point.current_u for point in payload.grid], dtype=np.float32)
    current_v = np.asarray([point.current_v for point in payload.grid], dtype=np.float32)
    wind_u = np.asarray([point.wind_u for point in payload.grid], dtype=np.float32)
    wind_v = np.asarray([point.wind_v for point in payload.grid], dtype=np.float32)
    shoreline = np.asarray([point.shoreline_proximity for point in payload.grid], dtype=np.float32)
    restricted = np.asarray([1.0 if point.restricted else 0.0 for point in payload.grid], dtype=np.float32)
    shoreline_boost = 0.85 + (shoreline * 0.45)
    stokes_u = wind_u * payload.stokes_drift_factor * shoreline_boost
    stokes_v = wind_v * payload.stokes_drift_factor * shoreline_boost
    return PreparedGrid(
        cell_ids=tuple(point.cell_id for point in payload.grid),
        latitudes=latitudes,
        longitudes=longitudes,
        current_u=current_u,
        current_v=current_v,
        wind_u=wind_u,
        wind_v=wind_v,
        shoreline=shoreline,
        restricted=restricted,
        stokes_u=stokes_u.astype(np.float32),
        stokes_v=stokes_v.astype(np.float32),
        source_weights=_source_weights(payload.grid, payload),
        lat_min=float(np.min(latitudes) - 0.02),
        lat_max=float(np.max(latitudes) + 0.02),
        lon_min=float(np.min(longitudes) - 0.02),
        lon_max=float(np.max(longitudes) + 0.02),
    )


def _nearest_indexes(latitudes: np.ndarray, longitudes: np.ndarray, prepared: PreparedGrid) -> np.ndarray:
    if latitudes.size == 0:
        return np.zeros((0,), dtype=np.intp)
    lat_diff = prepared.latitudes[None, :] - latitudes[:, None]
    lon_diff = prepared.longitudes[None, :] - longitudes[:, None]
    distances = (lat_diff * lat_diff) + (lon_diff * lon_diff)
    return np.argmin(distances, axis=1).astype(np.intp)


def _initialize_particles(prepared: PreparedGrid, payload: DriftBaselineInput, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    particle_count = payload.particles_per_member
    source_indexes = rng.choice(len(prepared.cell_ids), size=particle_count, replace=True, p=prepared.source_weights)
    jitter_km = 0.18
    latitudes = prepared.latitudes[source_indexes] + _km_to_lat(rng.normal(0.0, jitter_km, size=particle_count)).astype(np.float32)
    longitudes = prepared.longitudes[source_indexes] + _km_to_lon(
        rng.normal(0.0, jitter_km, size=particle_count).astype(np.float32),
        prepared.latitudes[source_indexes],
    ).astype(np.float32)
    return latitudes.astype(np.float32), longitudes.astype(np.float32)


def _simulate_member(
    prepared: PreparedGrid,
    payload: DriftBaselineInput,
    rng: np.random.Generator,
    windage_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    latitudes, longitudes = _initialize_particles(prepared, payload, rng)
    density_counts = np.zeros(len(prepared.cell_ids), dtype=np.int32)
    beach_counts = np.zeros(len(prepared.cell_ids), dtype=np.int32)
    beached_particles = 0
    diffusion_bias = min(0.12, payload.diffusion_sigma * 0.18)

    for _ in range(payload.horizon_hour):
        if latitudes.size == 0:
            break
        indexes = _nearest_indexes(latitudes, longitudes, prepared)
        eastward_km = (
            prepared.current_u[indexes]
            + (prepared.wind_u[indexes] * windage_fraction)
            + prepared.stokes_u[indexes]
            + rng.normal(0.0, payload.diffusion_sigma, size=indexes.shape[0]).astype(np.float32)
        )
        northward_km = (
            prepared.current_v[indexes]
            + (prepared.wind_v[indexes] * windage_fraction)
            + prepared.stokes_v[indexes]
            + rng.normal(0.0, payload.diffusion_sigma, size=indexes.shape[0]).astype(np.float32)
        )
        next_latitudes = np.clip(
            latitudes + _km_to_lat(northward_km).astype(np.float32),
            prepared.lat_min,
            prepared.lat_max,
        )
        next_longitudes = np.clip(
            longitudes + _km_to_lon(eastward_km.astype(np.float32), latitudes).astype(np.float32),
            prepared.lon_min,
            prepared.lon_max,
        )
        next_indexes = _nearest_indexes(next_latitudes, next_longitudes, prepared)
        beach_probability = np.maximum(
            0.0,
            (prepared.shoreline[next_indexes] - payload.beaching_threshold + 0.12)
            + (prepared.restricted[next_indexes] * 0.08)
            + diffusion_bias,
        )
        beached_mask = (beach_probability > 0.0) & (
            rng.random(size=next_indexes.shape[0]).astype(np.float32) < np.minimum(0.92, beach_probability)
        )
        if np.any(beached_mask):
            np.add.at(beach_counts, next_indexes[beached_mask], 1)
            beached_particles += int(np.sum(beached_mask))
        survivor_mask = ~beached_mask
        latitudes = next_latitudes[survivor_mask]
        longitudes = next_longitudes[survivor_mask]

    if latitudes.size:
        final_indexes = _nearest_indexes(latitudes, longitudes, prepared)
        np.add.at(density_counts, final_indexes, 1)

    return density_counts, beach_counts, beached_particles


def run_drift_baseline_ensemble(payload: DriftBaselineInput) -> DriftBaselineDiagnostics:
    prepared = _prepare_grid(payload)
    seed_rng = Random(_seed(payload))
    windage_fraction = _windage_fraction(payload)
    scale = len(payload.grid) * max(payload.source_strength, 0.1) / max(payload.particles_per_member, 1)
    density_members = np.zeros((payload.ensemble_members, len(prepared.cell_ids)), dtype=np.float32)
    beach_members = np.zeros((payload.ensemble_members, len(prepared.cell_ids)), dtype=np.float32)
    beached_particles = 0

    for ensemble_index in range(payload.ensemble_members):
        member_seed = seed_rng.randint(0, 10_000_000) + ensemble_index
        member_rng = np.random.default_rng(member_seed)
        density_counts, beach_counts, member_beached = _simulate_member(prepared, payload, member_rng, windage_fraction)
        beached_particles += member_beached
        density_members[ensemble_index] = density_counts.astype(np.float32) * scale
        beach_members[ensemble_index] = beach_counts.astype(np.float32) / max(payload.particles_per_member, 1)

    density_mean = np.mean(density_members, axis=0)
    density_spread = np.std(density_members, axis=0)
    beaching_mean = np.mean(beach_members, axis=0)
    density = {
        cell_id: round(max(0.0, float(density_mean[index])), 4)
        for index, cell_id in enumerate(prepared.cell_ids)
    }
    ensemble_spread = {
        cell_id: round(float(density_spread[index]), 4)
        for index, cell_id in enumerate(prepared.cell_ids)
    }
    beaching_fraction = {
        cell_id: round(float(beaching_mean[index]), 4)
        for index, cell_id in enumerate(prepared.cell_ids)
    }
    stokes_drift = {
        cell_id: (round(float(prepared.stokes_u[index]), 4), round(float(prepared.stokes_v[index]), 4))
        for index, cell_id in enumerate(prepared.cell_ids)
    }
    return DriftBaselineDiagnostics(
        density=density,
        ensemble_spread=ensemble_spread,
        beaching_fraction=beaching_fraction,
        stokes_drift=stokes_drift,
        windage_fraction=round(windage_fraction, 4),
        total_particles=payload.ensemble_members * payload.particles_per_member,
        beached_particles=beached_particles,
        ensemble_members=payload.ensemble_members,
    )


def run_drift_baseline(payload: DriftBaselineInput) -> dict[str, float]:
    return run_drift_baseline_ensemble(payload).density
