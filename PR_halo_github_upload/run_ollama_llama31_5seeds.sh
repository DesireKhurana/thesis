#!/usr/bin/env bash
set -euo pipefail

cd ~/thesis_halo/PR_halo
source .venv/bin/activate
export PATH="$HOME/ollama_usr/bin:$PATH"

export RUN_BACKENDS="ollama"
export OLLAMA_MODEL="llama3.1:8b"

export N_EPISODES=30
export RANDOM_PHASE_EPISODES=15
export P0=0.8
export IDENTITY_MODE="neutral"

mkdir -p logs/thesis_ollama_llama31_5seeds

MASTER_LOG="logs/thesis_ollama_llama31_5seeds/llama31_5seeds_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Ollama Llama 3.1 thesis matrix at $(date)" | tee -a "$MASTER_LOG"
echo "Model: llama3.1:8b" | tee -a "$MASTER_LOG"
echo "Seeds: 0 1 2 3 4" | tee -a "$MASTER_LOG"
echo "Episodes: 30" | tee -a "$MASTER_LOG"
echo "Conditions: neutral boss, neutral control, halo boss, halo control" | tee -a "$MASTER_LOG"

for seed in 0 1 2 3 4; do

  echo "" | tee -a "$MASTER_LOG"
  echo "===============================" | tee -a "$MASTER_LOG"
  echo "LLAMA3.1 SEED=$seed | neutral boss" | tee -a "$MASTER_LOG"
  echo "===============================" | tee -a "$MASTER_LOG"

  export SEED="$seed"
  export HALO_MODE="none"
  export HALO_WORKERS=""
  export HALO_STYLE="polite_structured"
  export ALL_RANDOM_CONTROL=0
  python main.py 2>&1 | tee -a "$MASTER_LOG"

  echo "" | tee -a "$MASTER_LOG"
  echo "===============================" | tee -a "$MASTER_LOG"
  echo "LLAMA3.1 SEED=$seed | neutral control" | tee -a "$MASTER_LOG"
  echo "===============================" | tee -a "$MASTER_LOG"

  export SEED="$seed"
  export HALO_MODE="none"
  export HALO_WORKERS=""
  export HALO_STYLE="polite_structured"
  export ALL_RANDOM_CONTROL=1
  python main.py 2>&1 | tee -a "$MASTER_LOG"

  echo "" | tee -a "$MASTER_LOG"
  echo "===============================" | tee -a "$MASTER_LOG"
  echo "LLAMA3.1 SEED=$seed | halo boss" | tee -a "$MASTER_LOG"
  echo "===============================" | tee -a "$MASTER_LOG"

  export SEED="$seed"
  export HALO_MODE="style"
  export HALO_WORKERS="W1,W2"
  export HALO_STYLE="polite_structured"
  export ALL_RANDOM_CONTROL=0
  python main.py 2>&1 | tee -a "$MASTER_LOG"

  echo "" | tee -a "$MASTER_LOG"
  echo "===============================" | tee -a "$MASTER_LOG"
  echo "LLAMA3.1 SEED=$seed | halo control" | tee -a "$MASTER_LOG"
  echo "===============================" | tee -a "$MASTER_LOG"

  export SEED="$seed"
  export HALO_MODE="style"
  export HALO_WORKERS="W1,W2"
  export HALO_STYLE="polite_structured"
  export ALL_RANDOM_CONTROL=1
  python main.py 2>&1 | tee -a "$MASTER_LOG"

done

echo "Finished Ollama Llama 3.1 thesis matrix at $(date)" | tee -a "$MASTER_LOG"
