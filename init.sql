CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS flux_observations (
    id SERIAL PRIMARY KEY,
    geom GEOMETRY(POINT, 4326),
    lat FLOAT NOT NULL,
    lon FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    co2_flux FLOAT,
    sst FLOAT,
    salinity FLOAT,
    wind_speed FLOAT,
    chl_a FLOAT,
    pco2_ocean FLOAT,
    anomaly_score FLOAT DEFAULT 0,
    source VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS idx_flux_geom ON flux_observations USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_flux_time ON flux_observations (timestamp);

CREATE TABLE IF NOT EXISTS flux_forecasts (
    id SERIAL PRIMARY KEY,
    lat FLOAT NOT NULL,
    lon FLOAT NOT NULL,
    forecast_time TIMESTAMP,
    valid_time TIMESTAMP,
    co2_flux FLOAT,
    confidence_low FLOAT,
    confidence_high FLOAT
);

CREATE TABLE IF NOT EXISTS anomalies (
    id UUID PRIMARY KEY,
    lat FLOAT,
    lon FLOAT,
    region_name VARCHAR(100),
    anomaly_score FLOAT,
    deviation_pct FLOAT,
    detected_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP,
    severity VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE
);
