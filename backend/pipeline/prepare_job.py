from __future__ import annotations

import argparse
import json

from .jobs import prepare_training_artifacts


def main():
    parser = argparse.ArgumentParser(description="Prepare HPC-friendly training artifacts for OceanPulse.")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--month-window", type=int, default=12)
    parser.add_argument("--resolution", default="2deg")
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()
    result = prepare_training_artifacts(
        {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "month_window": args.month_window,
            "resolution": args.resolution,
            "quick_test": args.quick_test,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
