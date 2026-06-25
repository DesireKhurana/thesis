#!/usr/bin/env bash
set -euo pipefail

cd ~/thesis_halo/PR_halo

PYTHON_BIN="/var/home/desire-khurana/venvs/pr_venv/bin/python"

export RUN_BACKENDS="groq"
export N_EPISODES=30
export RANDOM_PHASE_EPISODES=15
export P0=0.8
export SEED=1
export IDENTITY_MODE="neutral"
export HALO_STYLE="polite_structured"
export API_MAX_TOKENS=2048
export API_RATE_RETRIES=5

mkdir -p logs/thesis_groq_by_seed
LOG="logs/thesis_groq_by_seed/groq_seed1_remaining_$(date +%Y%m%d_%H%M%S).log"

run_one () {
  local halo_mode="$1"
  local halo_workers="$2"
  local control="$3"
  local label="$4"

  echo "" | tee -a "$LOG"
  echo "======================================" | tee -a "$LOG"
  echo "RUNNING: $label" | tee -a "$LOG"
  echo "SEED=$SEED HALO_MODE=$halo_mode HALO_WORKERS=$halo_workers ALL_RANDOM_CONTROL=$control" | tee -a "$LOG"
  echo "START: $(date)" | tee -a "$LOG"
  echo "======================================" | tee -a "$LOG"

  export HALO_MODE="$halo_mode"
  export HALO_WORKERS="$halo_workers"
  export ALL_RANDOM_CONTROL="$control"

  "$PYTHON_BIN" main.py 2>&1 | tee -a "$LOG"

  echo "FINISHED: $label at $(date)" | tee -a "$LOG"
}

run_one "none" ""       1 "Groq seed 1 neutral control"
run_one "style" "W1,W2" 0 "Groq seed 1 halo boss"
run_one "style" "W1,W2" 1 "Groq seed 1 halo control"

echo "" | tee -a "$LOG"
echo "GROQ SEED 1 REMAINING FINISHED at $(date)" | tee -a "$LOG"
