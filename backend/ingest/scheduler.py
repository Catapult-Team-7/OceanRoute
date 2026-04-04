from celery import Celery
from celery.schedules import crontab


app = Celery("oceanpulse")

app.conf.beat_schedule = {
    "refresh-demo-ocean-state": {"task": "ingest.refresh_demo_state", "schedule": crontab(minute="*/30")},
}
