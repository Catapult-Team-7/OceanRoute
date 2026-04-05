#!/bin/bash
#SBATCH --job-name=oceanpulse-publish
#SBATCH --output=logs/oceanpulse-publish-%j.out
#SBATCH --error=logs/oceanpulse-publish-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Catapult-2026}"
BACKEND_DIR="$ROOT_DIR/backend"

mkdir -p "$ROOT_DIR/logs"
cd "$BACKEND_DIR"

module purge

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python -m pipeline.publish_job --region "${REGION:-global}" --resolution "${RESOLUTION:-1deg}" ${TARGET_MONTH:+--date "$TARGET_MONTH"}

