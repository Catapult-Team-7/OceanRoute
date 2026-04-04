from db.demo_data import DemoOceanRepository


async def run_forecast(lat: float, lon: float, horizon_hours: int):
    repo = DemoOceanRepository()
    return repo.get_point_forecast(lat=lat, lon=lon, horizon_hours=horizon_hours)
