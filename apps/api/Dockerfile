FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OCEANROUTE_DATABASE_URL=postgresql+psycopg://oceanroute:oceanroute@db:5432/oceanroute
ENV OCEANROUTE_INGEST_MODE=auto
ENV OCEANROUTE_SCHEDULER_ENABLED=false

COPY apps/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY apps/api /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
