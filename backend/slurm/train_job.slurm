#!/bin/bash
#SBATCH --job-name=oceanpulse-train
#SBATCH --output=logs/oceanpulse-train-%j.out
#SBATCH --error=logs/oceanpulse-train-%j.err
#SBATCH --time=12:00:00
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

python -m pipeline.train_job --epochs "${EPOCHS:-80}" --month-window "${MONTH_WINDOW:-12}" --resolution "${RESOLUTION:-2deg}"

