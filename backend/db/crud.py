from .demo_data import DemoOceanRepository


async def get_flux_grid(repo: DemoOceanRepository, date: str | None, resolution: str, region: str):
    return repo.get_flux_grid(date=date, resolution=resolution, region=region)


async def get_recent_anomalies(repo: DemoOceanRepository, threshold: float, limit: int, date: str | None = None):
    return repo.get_recent_anomalies(threshold=threshold, limit=limit, date=date)


async def get_forecast(repo: DemoOceanRepository, lat: float, lon: float, horizon: int):
    return repo.get_point_forecast(lat=lat, lon=lon, horizon_hours=horizon)


async def get_stats(repo: DemoOceanRepository, date: str | None):
    return repo.get_global_stats(date=date)


async def get_history(repo: DemoOceanRepository, lat: float, lon: float, months: int):
    return repo.get_point_history(lat=lat, lon=lon, months=months)
