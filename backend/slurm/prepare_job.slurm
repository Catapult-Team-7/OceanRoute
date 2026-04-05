#!/bin/bash
#SBATCH --job-name=oceanpulse-prepare
#SBATCH --output=logs/oceanpulse-prepare-%j.out
#SBATCH --error=logs/oceanpulse-prepare-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Catapult-2026}"
BACKEND_DIR="$ROOT_DIR/backend"

mkdir -p "$ROOT_DIR/logs"
cd "$BACKEND_DIR"

module purge

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python -m pipeline.prepare_job --month-window "${MONTH_WINDOW:-12}" --resolution "${RESOLUTION:-2deg}"

