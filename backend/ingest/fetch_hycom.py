from datetime import datetime


def fetch_currents(date: datetime, lat_range=(-90, 90), lon_range=(-180, 180)):
    return {
        "date": date.isoformat(),
        "lat_range": lat_range,
        "lon_range": lon_range,
        "message": "HYCOM ingestion placeholder for hackathon demo build.",
    }
