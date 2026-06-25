#!/usr/bin/env bash
set -euo pipefail

cd ~/thesis_halo/PR_halo

PY="/var/home/desire-khurana/venvs/pr_venv/bin/python"

export RUN_BACKENDS="deepseek"
export N_EPISODES=30
export RANDOM_PHASE_EPISODES=15
export P0=0.8
export IDENTITY_MODE="neutral"
export HALO_STYLE="polite_structured"
export API_MAX_TOKENS=2048
export API_RATE_RETRIES=5

mkdir -p logs/thesis_deepseek_overnight
LOG="logs/thesis_deepseek_overnight/seed2_remaining_seed3_$(date +%Y%m%d_%H%M%S).log"

run_one () {
  local seed="$1"
  local halo_mode="$2"
  local halo_workers="$3"
  local control="$4"
  local label="$5"

  echo ""
  echo "======================================"
  echo "RUNNING: $label"
  echo "SEED=$seed HALO_MODE=$halo_mode HALO_WORKERS=$halo_workers ALL_RANDOM_CONTROL=$control"
  echo "START: $(date)"
  echo "======================================"

  export SEED="$seed"
  export HALO_MODE="$halo_mode"
  export HALO_WORKERS="$halo_workers"
  export ALL_RANDOM_CONTROL="$control"

  "$PY" main.py 2>&1 | tee -a "$LOG"

  echo "FINISHED: $label at $(date)" | tee -a "$LOG"
}

run_one 2 "style" "W1,W2" 1 "DeepSeek seed 2 halo control"

run_one 3 "none" "" 0 "DeepSeek seed 3 neutral boss"
run_one 3 "none" "" 1 "DeepSeek seed 3 neutral control"
run_one 3 "style" "W1,W2" 0 "DeepSeek seed 3 halo boss"
run_one 3 "style" "W1,W2" 1 "DeepSeek seed 3 halo control"

echo "ALL REQUESTED RUNS FINISHED at $(date)" | tee -a "$LOG"
