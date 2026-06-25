#!/usr/bin/env bash
set -euo pipefail

cd ~/thesis_halo/PR_halo

PYTHON="/var/home/desire-khurana/venvs/pr_venv/bin/python"

export N_EPISODES=30
export RANDOM_PHASE_EPISODES=15
export P0=0.8
export SEED=4
export IDENTITY_MODE="neutral"

export API_MAX_TOKENS=2048
export API_RATE_RETRIES=5

mkdir -p logs/thesis_missing_seed4
MASTER_LOG="logs/thesis_missing_seed4/missing_seed4_$(date +%Y%m%d_%H%M%S).log"

run_one() {
  local backend="$1"
  local condition="$2"

  echo "" | tee -a "$MASTER_LOG"
  echo "=========================================" | tee -a "$MASTER_LOG"
  echo "RUNNING: backend=$backend seed=4 condition=$condition" | tee -a "$MASTER_LOG"
  echo "=========================================" | tee -a "$MASTER_LOG"

  export RUN_BACKENDS="$backend"

  if [ "$condition" = "neutral_boss" ]; then
    export HALO_MODE="none"
    export HALO_WORKERS=""
    export HALO_STYLE="polite_structured"
    export ALL_RANDOM_CONTROL=0

  elif [ "$condition" = "neutral_control" ]; then
    export HALO_MODE="none"
    export HALO_WORKERS=""
    export HALO_STYLE="polite_structured"
    export ALL_RANDOM_CONTROL=1

  elif [ "$condition" = "halo_boss" ]; then
    export HALO_MODE="style"
    export HALO_WORKERS="W1,W2"
    export HALO_STYLE="polite_structured"
    export ALL_RANDOM_CONTROL=0

  elif [ "$condition" = "halo_control" ]; then
    export HALO_MODE="style"
    export HALO_WORKERS="W1,W2"
    export HALO_STYLE="polite_structured"
    export ALL_RANDOM_CONTROL=1

  else
    echo "Unknown condition: $condition"
    exit 1
  fi

  "$PYTHON" main.py 2>&1 | tee -a "$MASTER_LOG"

  echo "COMPLETED: backend=$backend seed=4 condition=$condition" | tee -a "$MASTER_LOG"
}

for backend in deepseek groq; do
  run_one "$backend" "neutral_boss"
  run_one "$backend" "neutral_control"
  run_one "$backend" "halo_boss"
  run_one "$backend" "halo_control"
done

echo "" | tee -a "$MASTER_LOG"
echo "Finished missing seed 4 runs at $(date)" | tee -a "$MASTER_LOG"
