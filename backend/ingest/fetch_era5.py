from __future__ import annotations

import argparse
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import zipfile

import pandas as pd


logger = logging.getLogger("oceanpulse.ingest.era5")

DATASET_ID = "reanalysis-era5-single-levels-timeseries"
DEFAULT_OUTPUT_DIRECTORY = Path("~/OceanPulseData/ERA").expanduser()
DEFAULT_LATITUDE = 27.5
DEFAULT_LONGITUDE = -140.0
DEFAULT_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_surface_temperature",
    "mean_sea_level_pressure",
]
COLUMN_ALIASES = {
    "valid_time": "valid_time",
    "time": "valid_time",
    "date": "valid_time",
    "datetime": "valid_time",
    "10m_u_component_of_wind": "u10",
    "u10": "u10",
    "10m_v_component_of_wind": "v10",
    "v10": "v10",
    "sea_surface_temperature": "sst",
    "sst": "sst",
    "mean_sea_level_pressure": "msl",
    "msl": "msl",
}


def _default_start_date() -> str:
    today = datetime.now(UTC).date()
    return (today - timedelta(days=730)).isoformat()


def _default_end_date() -> str:
    return datetime.now(UTC).date().isoformat()


def _output_filename(latitude: float, longitude: float, start: str, end: str) -> str:
    lat_token = f"{latitude:.2f}".replace("-", "m").replace(".", "p")
    lon_token = f"{longitude:.2f}".replace("-", "m").replace(".", "p")
    return f"era5_timeseries_lat_{lat_token}_lon_{lon_token}_{start}_{end}.csv"


def _read_era5_frame(csv_path: Path) -> pd.DataFrame:
    if zipfile.is_zipfile(csv_path):
        with zipfile.ZipFile(csv_path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                raise RuntimeError(f"ERA5 archive {csv_path} does not contain a CSV file.")
            with archive.open(csv_members[0]) as handle:
                try:
                    return pd.read_csv(handle, low_memory=False)
                except UnicodeDecodeError:
                    handle.close()
            with archive.open(csv_members[0]) as handle:
                return pd.read_csv(handle, low_memory=False, encoding="latin1")

    try:
        return pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, encoding="latin1")


def _normalize_era5_csv(csv_path: Path) -> list[str]:
    frame = _read_era5_frame(csv_path)
    if frame.empty:
        raise RuntimeError(f"ERA5 download completed but {csv_path} is empty.")

    rename_map: dict[str, str] = {}
    for column in frame.columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        if normalized in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[normalized]
    frame = frame.rename(columns=rename_map)

    if "valid_time" not in frame.columns:
        raise RuntimeError(
            "Downloaded ERA5 CSV is missing a recognizable time column. "
            f"Columns found: {list(frame.columns)}"
        )

    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["valid_time"]).sort_values("valid_time").reset_index(drop=True)

    for column in ("u10", "v10", "sst", "msl"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    required_columns = {"valid_time", "u10", "v10"}
    if not required_columns.issubset(frame.columns):
        raise RuntimeError(
            "Downloaded ERA5 CSV does not contain the columns required by training. "
            f"Needed: {sorted(required_columns)}. Found: {list(frame.columns)}"
        )

    output_columns = ["valid_time"] + [column for column in ("u10", "v10", "sst", "msl") if column in frame.columns]
    frame[output_columns].to_csv(csv_path, index=False)
    return output_columns


def _build_request(*, latitude: float, longitude: float, start_date: str, end_date: str, variables: list[str]) -> dict:
    # Inference from the official CDS timeseries dataset form/API pattern:
    # use a single location object plus a date range and CSV output.
    return {
        "variable": variables,
        "location": {"latitude": latitude, "longitude": longitude},
        "date": [f"{start_date}/{end_date}"],
        "data_format": "csv",
    }


def download_era5_timeseries(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    output_directory: Path,
    filename: str | None = None,
    variables: list[str] | None = None,
) -> Path:
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover - dependency branch
        raise RuntimeError("`cdsapi` is not installed. Install backend/requirements-ml-ingest.txt first.") from exc

    output_directory.mkdir(parents=True, exist_ok=True)
    target_path = output_directory / (filename or _output_filename(latitude, longitude, start_date, end_date))
    request = _build_request(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        variables=variables or DEFAULT_VARIABLES,
    )

    logger.info(
        "downloading_era5_timeseries dataset=%s lat=%s lon=%s start=%s end=%s target=%s",
        DATASET_ID,
        latitude,
        longitude,
        start_date,
        end_date,
        target_path,
    )
    client = cdsapi.Client()
    client.retrieve(DATASET_ID, request, str(target_path))
    columns = _normalize_era5_csv(target_path)
    logger.info("era5_download_complete path=%s columns=%s", target_path, columns)
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5 single-point time-series CSVs for OceanPulse training."
    )
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE, help="Point latitude for the ERA5 request.")
    parser.add_argument(
        "--longitude", type=float, default=DEFAULT_LONGITUDE, help="Point longitude for the ERA5 request."
    )
    parser.add_argument(
        "--start-date",
        default=_default_start_date(),
        help="Start date in YYYY-MM-DD format. Defaults to roughly two years ago.",
    )
    parser.add_argument(
        "--end-date",
        default=_default_end_date(),
        help="End date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help="Directory where the normalized CSV will be written.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional output filename. Defaults to a location/date-based filename.",
    )
    parser.add_argument(
        "--variable",
        action="append",
        dest="variables",
        help="Optional CDS variable override. Repeat to pass multiple variables.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    output_path = download_era5_timeseries(
        latitude=args.latitude,
        longitude=args.longitude,
        start_date=args.start_date,
        end_date=args.end_date,
        output_directory=Path(args.output_dir).expanduser(),
        filename=args.filename,
        variables=args.variables,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
