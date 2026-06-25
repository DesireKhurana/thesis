#!/usr/bin/env bash
set -u

cd ~/thesis_halo/PR_halo
source .venv/bin/activate

export PATH="$HOME/ollama_usr/bin:$PATH"

GPU_ID=0
MODEL="llama3.1:8b"
MODEL_SAFE="llama3.1-8b"

export RUN_BACKENDS="ollama"
export OLLAMA_MODEL="$MODEL"

export N_EPISODES=30
export RANDOM_PHASE_EPISODES=15
export P0=0.8
export IDENTITY_MODE="neutral"

mkdir -p logs/llama31_5seeds_overnight

MASTER_LOG="logs/llama31_5seeds_overnight/master_$(date +%Y%m%d_%H%M%S).log"
STATUS_FILE="logs/llama31_5seeds_overnight/status_$(date +%Y%m%d_%H%M%S).csv"
OLLAMA_LOG="$HOME/ollama_server_gpu${GPU_ID}_llama31.log"

echo "condition,seed,status,run_dir" > "$STATUS_FILE"

echo "Starting llama3.1:8b matrix at $(date)" | tee -a "$MASTER_LOG"
echo "GPU_ID=$GPU_ID" | tee -a "$MASTER_LOG"
echo "MODEL=$MODEL" | tee -a "$MASTER_LOG"
echo "Seeds: 0 1 2 3 4" | tee -a "$MASTER_LOG"
echo "Conditions: neutral boss/control, halo boss/control" | tee -a "$MASTER_LOG"

echo "Restarting Ollama on GPU $GPU_ID..." | tee -a "$MASTER_LOG"
pkill -u "$USER" -f "ollama serve" 2>/dev/null || true
CUDA_VISIBLE_DEVICES="$GPU_ID" nohup ollama serve > "$OLLAMA_LOG" 2>&1 &
sleep 8

echo "Checking Ollama models..." | tee -a "$MASTER_LOG"
ollama list | tee -a "$MASTER_LOG"

echo "Testing model..." | tee -a "$MASTER_LOG"
ollama run "$MODEL" "Reply with exactly: READY" | tee -a "$MASTER_LOG"

latest_matching_run() {
  local pattern="$1"
  find runs/ollama -maxdepth 1 -type d -name "$pattern" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

is_complete_run() {
  local run_dir="$1"
  [ -n "$run_dir" ] || return 1
  [ -f "$run_dir/episodes.jsonl" ] || return 1
  [ -f "$run_dir/stage_eval.jsonl" ] || return 1

  ep_count=$(wc -l < "$run_dir/episodes.jsonl")
  stage_count=$(wc -l < "$run_dir/stage_eval.jsonl")

  [ "$ep_count" -eq "$N_EPISODES" ] && [ "$stage_count" -ge 2 ]
}

run_one() {
  local seed="$1"
  local halo_mode="$2"
  local halo_workers="$3"
  local control="$4"
  local label="$5"

  export SEED="$seed"
  export HALO_MODE="$halo_mode"
  export HALO_WORKERS="$halo_workers"
  export HALO_STYLE="polite_structured"
  export ALL_RANDOM_CONTROL="$control"

  if [ "$halo_mode" = "none" ]; then
    if [ "$control" = "1" ]; then
      condition_label="neutral_control_seed${seed}"
    else
      condition_label="neutral_boss_seed${seed}"
    fi
  else
    if [ "$control" = "1" ]; then
      condition_label="halo-style_polite_structured_control_seed${seed}"
    else
      condition_label="halo-style_polite_structured_boss_seed${seed}"
    fi
  fi

  pattern="${MODEL_SAFE}_${condition_label}_*"
  existing_run=$(latest_matching_run "$pattern")

  if is_complete_run "$existing_run"; then
    echo "SKIP complete: $label seed=$seed -> $existing_run" | tee -a "$MASTER_LOG"
    echo "$label,$seed,skipped_complete,$existing_run" >> "$STATUS_FILE"
    return 0
  fi

  run_log="logs/llama31_5seeds_overnight/${condition_label}_$(date +%Y%m%d_%H%M%S).log"

  echo "" | tee -a "$MASTER_LOG"
  echo "==================================================" | tee -a "$MASTER_LOG"
  echo "START $label seed=$seed at $(date)" | tee -a "$MASTER_LOG"
  echo "condition_label=$condition_label" | tee -a "$MASTER_LOG"
  echo "HALO_MODE=$HALO_MODE HALO_WORKERS=$HALO_WORKERS ALL_RANDOM_CONTROL=$ALL_RANDOM_CONTROL" | tee -a "$MASTER_LOG"
  echo "Run log: $run_log" | tee -a "$MASTER_LOG"
  echo "==================================================" | tee -a "$MASTER_LOG"

  python main.py > "$run_log" 2>&1
  status=$?

  new_run=$(latest_matching_run "$pattern")

  echo "END $label seed=$seed status=$status at $(date)" | tee -a "$MASTER_LOG"
  echo "Run dir: $new_run" | tee -a "$MASTER_LOG"
  tail -n 25 "$run_log" | tee -a "$MASTER_LOG"

  if [ "$status" -eq 0 ] && is_complete_run "$new_run"; then
    echo "$label,$seed,complete,$new_run" >> "$STATUS_FILE"
  else
    echo "$label,$seed,failed_or_incomplete,$new_run" >> "$STATUS_FILE"
  fi
}

for seed in 0 1 2 3 4; do
  run_one "$seed" "none"  ""      "0" "neutral_boss"
  run_one "$seed" "none"  ""      "1" "neutral_control"
  run_one "$seed" "style" "W1,W2" "0" "halo_boss"
  run_one "$seed" "style" "W1,W2" "1" "halo_control"
done

echo "" | tee -a "$MASTER_LOG"
echo "Finished llama3.1:8b matrix at $(date)" | tee -a "$MASTER_LOG"
echo "Status file: $STATUS_FILE" | tee -a "$MASTER_LOG"
cat "$STATUS_FILE" | tee -a "$MASTER_LOG"
