from datetime import datetime


def fetch_sst(date: datetime):
    return {"date": date.isoformat(), "message": "SST ingestion placeholder for hackathon demo build."}
