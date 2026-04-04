from __future__ import annotations

from app.schemas import ForecastSnapshot


def forecast_to_geojson(snapshot: ForecastSnapshot) -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "metadata": {
            "run_id": snapshot.run_id,
            "region_id": snapshot.region.id,
            "region_name": snapshot.region.name,
            "generated_at": snapshot.generated_at.isoformat(),
            "horizon_hours": snapshot.horizon_hours,
            "source_mode_requested": snapshot.source_mode_requested,
            "source_mode_used": snapshot.source_mode_used,
            "is_fallback": snapshot.is_fallback,
            "is_stale": snapshot.is_stale,
            "age_minutes": snapshot.age_minutes,
            "stale_after_minutes": snapshot.stale_after_minutes,
        },
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item.lon, item.lat]},
                "properties": {
                    "cell_id": item.cell_id,
                    "debris_class": item.debris_class,
                    "valid_at": item.valid_at.isoformat(),
                    "horizon_hour": item.horizon_hour,
                    "probability": item.probability,
                    "baseline_density": item.baseline_density,
                    "ensemble_spread": item.ensemble_spread,
                    "beaching_fraction": item.beaching_fraction,
                    "stokes_drift_u": item.stokes_drift_u,
                    "stokes_drift_v": item.stokes_drift_v,
                    "windage_fraction": item.windage_fraction,
                    "expected_kg_min": item.expected_kg_min,
                    "expected_kg_max": item.expected_kg_max,
                    "confidence": item.confidence,
                    "uncertainty": item.uncertainty,
                    "beaching_risk": item.beaching_risk,
                },
            }
            for item in snapshot.steps
        ],
    }


def build_pdf_brief_bytes(snapshot: ForecastSnapshot) -> bytes:
    top_lines = []
    for hotspot in snapshot.top_hotspots[:4]:
        top_lines.append(
            f"{hotspot.cell_id} ({hotspot.debris_class}) h+{hotspot.horizon_hour}: {hotspot.expected_kg_min}-{hotspot.expected_kg_max} kg"
        )
    summary_text = "\\n".join(top_lines) if top_lines else "No hotspots available."
    text = (
        f"OceanRoute Mission Brief\\n"
        f"Run: {snapshot.run_id}\\n"
        f"Region: {snapshot.region.name} ({snapshot.pilot_region})\\n"
        f"Generated: {snapshot.generated_at.isoformat()}\\n"
        f"Horizon: {snapshot.horizon_hours}h\\n"
        f"Source: {snapshot.source_mode_requested}->{snapshot.source_mode_used}\\n"
        f"Fallback: {snapshot.is_fallback}\\n"
        f"Stale: {snapshot.is_stale}\\n"
        f"Top Hotspots:\\n{summary_text}"
    )
    stream = f"BT /F1 11 Tf 42 770 Td ({text}) Tj ET"
    pdf = (
        "%PDF-1.4\n"
        "1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        "2 0 obj<< /Type /Pages /Count 1 /Kids [3 0 R] >>endobj\n"
        "3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources<< /Font<< /F1 4 0 R >> >> /Contents 5 0 R >>endobj\n"
        "4 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
        f"5 0 obj<< /Length {len(stream)} >>stream\n{stream}\nendstream endobj\n"
        "xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000244 00000 n \n0000000314 00000 n \n"
        "trailer<< /Root 1 0 R /Size 6 >>\nstartxref\n451\n%%EOF"
    )
    return pdf.encode("utf-8")
