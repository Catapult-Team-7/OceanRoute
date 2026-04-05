#!/bin/bash
#SBATCH --job-name=oceanpulse-full
#SBATCH --output=logs/oceanpulse-full-%j.out
#SBATCH --error=logs/oceanpulse-full-%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Catapult-2026}"
BACKEND_DIR="$ROOT_DIR/backend"

mkdir -p "$ROOT_DIR/logs"
cd "$BACKEND_DIR"

module purge

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python -m pipeline.prepare_job --month-window "${MONTH_WINDOW:-12}" --resolution "${TRAIN_RESOLUTION:-2deg}"
python -m pipeline.train_job --epochs "${EPOCHS:-80}" --month-window "${MONTH_WINDOW:-12}" --resolution "${TRAIN_RESOLUTION:-2deg}"
python -m pipeline.publish_job --region "${PUBLISH_REGION:-global}" --resolution "${PUBLISH_RESOLUTION:-1deg}" ${TARGET_MONTH:+--date "$TARGET_MONTH"}

