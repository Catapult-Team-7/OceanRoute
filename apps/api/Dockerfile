FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SEASWEEP_DATABASE_URL=postgresql+psycopg://seasweep:seasweep@db:5432/seasweep
ENV SEASWEEP_INGEST_MODE=auto
ENV SEASWEEP_SCHEDULER_ENABLED=false

COPY apps/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY apps/api /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
