#!/bin/bash
#SBATCH --job-name=noitarl-test-quick
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p slurm/logs

CPUS="${SLURM_CPUS_PER_TASK:-8}"
RESERVE="${NOITARL_CPU_RESERVE:-1}"
export NOITARL_TEST_N_WORKERS="${NOITARL_TEST_N_WORKERS:-$((CPUS - RESERVE))}"
export AGENT__DEVICE="${AGENT__DEVICE:-cpu}"

echo "repo=$ROOT cpus=$CPUS test_n_workers=$NOITARL_TEST_N_WORKERS device=$AGENT__DEVICE"

uv sync --group dev
uv run pytest
