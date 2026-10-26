#!/bin/bash
# RUN__NAME=... AGENT__TOTAL_STEPS=... sbatch slurm/train.sbatch
# GPU jobs default to 8 workers (RolloutStorage lives on GPU). Raise NOITARL_GPU_N_WORKERS or AGENT__N_MINIBATCHES on larger GPUs.
# Override SLURM limits: sbatch --time=1-00:00:00 --partition=... slurm/train.sbatch
#SBATCH --job-name=noitarl-train
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p slurm/logs

CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-32}}"
RESERVE="${NOITARL_CPU_RESERVE:-2}"
DEVICE="${AGENT__DEVICE:-cuda}"

export RUN__NAME="${RUN__NAME:-slurm-train}"
export AGENT__TOTAL_STEPS="${AGENT__TOTAL_STEPS:-4096}"
export AGENT__DEVICE="$DEVICE"
if [[ -z "${AGENT__N_WORKERS:-}" ]]; then
  if [[ "$DEVICE" == "cpu" ]]; then
    export AGENT__N_WORKERS="$((CPUS - RESERVE))"
  else
    export AGENT__N_WORKERS="${NOITARL_GPU_N_WORKERS:-8}"
  fi
fi

echo "repo=$ROOT"
echo "run_name=$RUN__NAME total_steps=$AGENT__TOTAL_STEPS device=$AGENT__DEVICE n_workers=$AGENT__N_WORKERS cpus=$CPUS"
echo "rollout_batch=$((AGENT__N_WORKERS * ${AGENT__N_STEPS:-512})) ppo_minibatch=$(( (AGENT__N_WORKERS * ${AGENT__N_STEPS:-512}) / ${AGENT__N_MINIBATCHES:-8} ))"

uv sync
uv run noitarl train
