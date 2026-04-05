from __future__ import annotations

import argparse
import json
import sys

from ingest.real_training_data import RealDataLoadError
from .jobs import publish_verified_map


def main():
    parser = argparse.ArgumentParser(description="Publish a verified OceanPulse map bundle for serving.")
    parser.add_argument("--date", default=None)
    parser.add_argument("--resolution", default="1deg")
    parser.add_argument("--region", default="global")
    args = parser.parse_args()
    try:
        result = publish_verified_map(date=args.date, resolution=args.resolution, region=args.region)
    except RealDataLoadError as exc:
        result = {
            "status": "blocked",
            "message": str(exc),
            "metadata": {
                "verified_map": False,
                "map_source": "unpublished_real_grid",
                "source_summary": f"Verified map publish is blocked: {exc}",
            },
        }
        print(json.dumps(result, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
