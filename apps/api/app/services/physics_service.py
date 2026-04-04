from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sqrt
from random import Random

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


@dataclass
class Particle:
    lat: float
    lon: float


def _seed(payload: DriftBaselineInput) -> int:
    return 7000 + payload.horizon_hour + len(payload.grid) + int(payload.source_strength * 1000)


def _windage_fraction(payload: DriftBaselineInput) -> float:
    class_fraction = float(DEBRIS_CLASS_METADATA[payload.debris_class]["windage_factor"])
    return max(0.0, payload.windage_factor + class_fraction)


def _nearest_point(lat: float, lon: float, grid: list[GridPoint]) -> GridPoint:
    best_point = grid[0]
    best_distance = float("inf")
    for point in grid:
        distance = ((point.lat - lat) ** 2) + ((point.lon - lon) ** 2)
        if distance < best_distance:
            best_distance = distance
            best_point = point
    return best_point


def _km_to_lat(delta_km: float) -> float:
    return delta_km / 111.0


def _km_to_lon(delta_km: float, latitude: float) -> float:
    scale = max(0.2, cos(radians(latitude)))
    return delta_km / (111.0 * scale)


def _stokes_drift(point: GridPoint, payload: DriftBaselineInput) -> tuple[float, float]:
    shoreline_boost = 0.85 + (point.shoreline_proximity * 0.45)
    return (
        point.wind_u * payload.stokes_drift_factor * shoreline_boost,
        point.wind_v * payload.stokes_drift_factor * shoreline_boost,
    )


def _source_weight(point: GridPoint, payload: DriftBaselineInput) -> float:
    current_speed = sqrt((point.current_u**2) + (point.current_v**2))
    wind_speed = sqrt((point.wind_u**2) + (point.wind_v**2))
    shoreline_bias = 0.55 + (point.shoreline_proximity * 1.1)
    drift_bias = 0.2 + (current_speed * 0.18) + (wind_speed * 0.015)
    class_bias = 0.12 if payload.debris_class == "high" else 0.05
    restriction_penalty = 0.82 if point.restricted else 1.0
    return max(0.05, (shoreline_bias + drift_bias + class_bias) * restriction_penalty)


def _initialize_particles(payload: DriftBaselineInput, rng: Random) -> list[Particle]:
    weighted_points = [(point, _source_weight(point, payload)) for point in payload.grid]
    total_weight = sum(weight for _, weight in weighted_points)
    particles: list[Particle] = []
    jitter_km = 0.18
    for point, weight in weighted_points:
        target_count = max(1, round((weight / max(total_weight, 1e-6)) * payload.particles_per_member))
        for _ in range(target_count):
            lat = point.lat + _km_to_lat(rng.gauss(0.0, jitter_km))
            lon = point.lon + _km_to_lon(rng.gauss(0.0, jitter_km), point.lat)
            particles.append(Particle(lat=lat, lon=lon))
    return particles[: payload.particles_per_member]


def _clamp_to_bbox(lat: float, lon: float, payload: DriftBaselineInput) -> tuple[float, float]:
    latitudes = [point.lat for point in payload.grid]
    longitudes = [point.lon for point in payload.grid]
    return (
        min(max(lat, min(latitudes) - 0.02), max(latitudes) + 0.02),
        min(max(lon, min(longitudes) - 0.02), max(longitudes) + 0.02),
    )


def _simulate_member(
    payload: DriftBaselineInput,
    rng: Random,
    windage_fraction: float,
) -> tuple[dict[str, int], dict[str, int], int]:
    particles = _initialize_particles(payload, rng)
    density_counts = {point.cell_id: 0 for point in payload.grid}
    beach_counts = {point.cell_id: 0 for point in payload.grid}
    beached = 0

    for _ in range(payload.horizon_hour):
        survivors: list[Particle] = []
        for particle in particles:
            point = _nearest_point(particle.lat, particle.lon, payload.grid)
            stokes_u, stokes_v = _stokes_drift(point, payload)
            eastward_km = point.current_u + (point.wind_u * windage_fraction) + stokes_u + rng.gauss(0.0, payload.diffusion_sigma)
            northward_km = point.current_v + (point.wind_v * windage_fraction) + stokes_v + rng.gauss(0.0, payload.diffusion_sigma)
            next_lat = particle.lat + _km_to_lat(northward_km)
            next_lon = particle.lon + _km_to_lon(eastward_km, particle.lat)
            next_lat, next_lon = _clamp_to_bbox(next_lat, next_lon, payload)
            next_point = _nearest_point(next_lat, next_lon, payload.grid)

            beach_probability = max(
                0.0,
                (next_point.shoreline_proximity - payload.beaching_threshold + 0.12)
                + (0.08 if next_point.restricted else 0.0)
                + min(0.12, payload.diffusion_sigma * 0.18),
            )
            if beach_probability > 0 and rng.random() < min(0.92, beach_probability):
                beach_counts[next_point.cell_id] += 1
                beached += 1
                continue

            survivors.append(Particle(lat=next_lat, lon=next_lon))
        particles = survivors
        if not particles:
            break

    for particle in particles:
        point = _nearest_point(particle.lat, particle.lon, payload.grid)
        density_counts[point.cell_id] += 1

    return density_counts, beach_counts, beached


def run_drift_baseline_ensemble(payload: DriftBaselineInput) -> DriftBaselineDiagnostics:
    rng = Random(_seed(payload))
    windage_fraction = _windage_fraction(payload)
    scale = len(payload.grid) * max(payload.source_strength, 0.1) / max(payload.particles_per_member, 1)
    density_members = {point.cell_id: [] for point in payload.grid}
    beach_members = {point.cell_id: [] for point in payload.grid}
    beached_particles = 0

    for ensemble_index in range(payload.ensemble_members):
        member_rng = Random(rng.randint(0, 10_000_000) + ensemble_index)
        density_counts, beach_counts, member_beached = _simulate_member(payload, member_rng, windage_fraction)
        beached_particles += member_beached
        for point in payload.grid:
            density_members[point.cell_id].append(density_counts[point.cell_id] * scale)
            beach_members[point.cell_id].append(beach_counts[point.cell_id] / max(payload.particles_per_member, 1))

    density: dict[str, float] = {}
    ensemble_spread: dict[str, float] = {}
    beaching_fraction: dict[str, float] = {}
    stokes_drift = {point.cell_id: _stokes_drift(point, payload) for point in payload.grid}
    for point in payload.grid:
        member_values = density_members[point.cell_id]
        member_mean = sum(member_values) / len(member_values)
        member_variance = sum((value - member_mean) ** 2 for value in member_values) / len(member_values)
        density[point.cell_id] = round(max(0.0, member_mean), 4)
        ensemble_spread[point.cell_id] = round(member_variance**0.5, 4)
        beaching_fraction[point.cell_id] = round(sum(beach_members[point.cell_id]) / len(beach_members[point.cell_id]), 4)

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
