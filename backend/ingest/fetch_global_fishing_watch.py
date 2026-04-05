from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_GFW_API_ROOT = "https://gateway.api.globalfishingwatch.org/v3"
DEFAULT_GFW_VESSEL_DATASET = "public-global-vessel-identity:latest"
DEFAULT_GFW_PORT_VISITS_DATASET = "public-global-port-visits-events:latest"


class GlobalFishingWatchError(RuntimeError):
    pass


def _headers(token: str | None = None) -> dict[str, str]:
    api_token = (token or os.getenv("GLOBAL_FISHING_WATCH_TOKEN", "")).strip()
    if not api_token:
        raise GlobalFishingWatchError("GLOBAL_FISHING_WATCH_TOKEN is not configured.")
    return {"Authorization": f"Bearer {api_token}"}


def search_vessels(query: str, *, token: str | None = None, limit: int = 10) -> dict[str, Any]:
    response = requests.get(
        f"{DEFAULT_GFW_API_ROOT}/vessels/search",
        headers=_headers(token),
        params={
            "query": query,
            "datasets[0]": DEFAULT_GFW_VESSEL_DATASET,
            "limit": max(1, min(limit, 50)),
        },
        timeout=25,
    )
    response.raise_for_status()
    return response.json()


def get_port_visits(
    *,
    start_date: str,
    end_date: str,
    token: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    response = requests.get(
        f"{DEFAULT_GFW_API_ROOT}/events",
        headers=_headers(token),
        params={
            "datasets[0]": DEFAULT_GFW_PORT_VISITS_DATASET,
            "startDate": start_date,
            "endDate": end_date,
            "limit": max(1, min(limit, 100)),
        },
        timeout=25,
    )
    response.raise_for_status()
    return response.json()
