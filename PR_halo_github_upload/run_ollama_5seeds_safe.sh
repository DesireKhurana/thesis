#!/usr/bin/env bash
set -euo pipefail

cd ~/thesis_halo/PR_halo

export PATH="$HOME/ollama_usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/ollama_usr/lib/ollama:${LD_LIBRARY_PATH:-}"

PYTHON="/var/home/desire-khurana/venvs/pr_venv/bin/python"

export RUN_BACKENDS="ollama"
export OLLAMA_MODEL="qwen2.5:7b-instruct"

export N_EPISODES=30
export RANDOM_PHASE_EPISODES=15
export P0=0.8
export IDENTITY_MODE="neutral"

mkdir -p logs/thesis_ollama_5seeds
mkdir -p results

DONE_FILE="results/ollama_completed_runs.tsv"
MASTER_LOG="logs/thesis_ollama_5seeds/ollama_5seeds_$(date +%Y%m%d_%H%M%S).log"

touch "$DONE_FILE"

echo "Starting Ollama thesis matrix at $(date)" | tee -a "$MASTER_LOG"
echo "Model: qwen2.5:7b-instruct" | tee -a "$MASTER_LOG"
echo "Seeds: 0 1 2 3 4" | tee -a "$MASTER_LOG"
echo "Conditions: neutral boss, neutral control, halo boss, halo control" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

check_ollama() {
  echo "Checking Ollama server..." | tee -a "$MASTER_LOG"
  curl -s http://127.0.0.1:11434/api/tags | head | tee -a "$MASTER_LOG"
  ollama list | tee -a "$MASTER_LOG"
}

is_done() {
  local seed="$1"
  local condition="$2"
  grep -q "^${seed}[[:space:]]${condition}[[:space:]]" "$DONE_FILE"
}

mark_done() {
  local seed="$1"
  local condition="$2"
  local run_dir="$3"
  printf "%s\t%s\t%s\t%s\n" "$seed" "$condition" "$(date +%Y-%m-%d_%H:%M:%S)" "$run_dir" >> "$DONE_FILE"
}

validate_latest_run() {
  local pattern="$1"
  local final_stage="$2"

  local run_dir
  run_dir=$(find runs/ollama -maxdepth 1 -type d -name "$pattern" -printf "%T@ %p\n" | sort -nr | head -n 1 | cut -d' ' -f2-)

  if [ -z "$run_dir" ]; then
    echo "ERROR: Could not find run folder for pattern: $pattern" >&2
    exit 1
  fi

  echo "Latest run dir: $run_dir" | tee -a "$MASTER_LOG" >&2

  "$PYTHON" - "$run_dir" "$final_stage" >&2 <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
final_stage = sys.argv[2]

cfg_path = run_dir / "run_config.json"
episodes_path = run_dir / "episodes.jsonl"
stage_path = run_dir / "stage_eval.jsonl"

if not cfg_path.exists():
    raise SystemExit(f"Missing run_config.json: {run_dir}")

if not episodes_path.exists():
    raise SystemExit(f"Missing episodes.jsonl: {run_dir}")

if not stage_path.exists():
    raise SystemExit(f"Missing stage_eval.jsonl: {run_dir}")

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

if int(cfg.get("n_episodes", -1)) != 30:
    raise SystemExit(f"Wrong n_episodes in {run_dir}: {cfg.get('n_episodes')}")

if int(cfg.get("random_phase_episodes", -1)) != 15:
    raise SystemExit(f"Wrong random_phase_episodes in {run_dir}: {cfg.get('random_phase_episodes')}")

episodes = [line for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(episodes) != 30:
    raise SystemExit(f"Expected 30 episodes, found {len(episodes)} in {run_dir}")

stages = []
for line in stage_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        stages.append(json.loads(line))

stage_names = [s.get("stage") for s in stages]

if "random" not in stage_names:
    raise SystemExit(f"Missing random stage in {run_dir}")

if final_stage not in stage_names:
    raise SystemExit(f"Missing final stage {final_stage} in {run_dir}")

print("VALIDATED:", run_dir)
PY

  echo "$run_dir"
}

run_one() {
  local seed="$1"
  local condition="$2"

  if is_done "$seed" "$condition"; then
    echo "SKIP: seed=$seed condition=$condition already completed" | tee -a "$MASTER_LOG"
    return
  fi

  echo "" | tee -a "$MASTER_LOG"
  echo "=========================================" | tee -a "$MASTER_LOG"
  echo "RUNNING: seed=$seed condition=$condition" | tee -a "$MASTER_LOG"
  echo "=========================================" | tee -a "$MASTER_LOG"

  export SEED="$seed"

  if [ "$condition" = "neutral_boss" ]; then
    export HALO_MODE="none"
    export HALO_WORKERS=""
    export HALO_STYLE="polite_structured"
    export ALL_RANDOM_CONTROL=0
    local pattern="qwen2.5-7b-instruct_neutral_boss_seed${seed}_*"
    local final_stage="boss"

  elif [ "$condition" = "neutral_control" ]; then
    export HALO_MODE="none"
    export HALO_WORKERS=""
    export HALO_STYLE="polite_structured"
    export ALL_RANDOM_CONTROL=1
    local pattern="qwen2.5-7b-instruct_neutral_control_seed${seed}_*"
    local final_stage="random_control"

  elif [ "$condition" = "halo_boss" ]; then
    export HALO_MODE="style"
    export HALO_WORKERS="W1,W2"
    export HALO_STYLE="polite_structured"
    export ALL_RANDOM_CONTROL=0
    local pattern="qwen2.5-7b-instruct_halo-style_polite_structured_boss_seed${seed}_*"
    local final_stage="boss"

  elif [ "$condition" = "halo_control" ]; then
    export HALO_MODE="style"
    export HALO_WORKERS="W1,W2"
    export HALO_STYLE="polite_structured"
    export ALL_RANDOM_CONTROL=1
    local pattern="qwen2.5-7b-instruct_halo-style_polite_structured_control_seed${seed}_*"
    local final_stage="random_control"

  else
    echo "Unknown condition: $condition"
    exit 1
  fi

  "$PYTHON" main.py 2>&1 | tee -a "$MASTER_LOG"

  run_dir=$(validate_latest_run "$pattern" "$final_stage")
  mark_done "$seed" "$condition" "$run_dir"

  echo "COMPLETED: seed=$seed condition=$condition" | tee -a "$MASTER_LOG"
}

check_ollama

for seed in 0 1 2 3 4; do
  run_one "$seed" "neutral_boss"
  run_one "$seed" "neutral_control"
  run_one "$seed" "halo_boss"
  run_one "$seed" "halo_control"
done

echo "" | tee -a "$MASTER_LOG"
echo "Finished Ollama thesis matrix at $(date)" | tee -a "$MASTER_LOG"
echo "Completed runs saved in: $DONE_FILE" | tee -a "$MASTER_LOG"
