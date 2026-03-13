#!/usr/bin/env bash
set -euo pipefail

# Expanded evaluation: 5 models, 40 scenarios (10 train / 30 eval), 3 episodes.
# Conditions: static, replay, rules (main) + static-no-normalize (ablation).
#
# Usage:
#   bash scripts/run_expanded_evaluation.sh
# Optional env:
#   OUTPUT_ROOT=results/raw_task_success_expanded_2026-03-10
#   ABLATION_OUTPUT_ROOT=results/raw_task_success_nonorm_2026-03-10
#   EPISODES=3
#   MODELS="or_gpt4o_mini or_kimi_k2 or_qwen3_32b or_gpt4o or_deepseek_v3"
#   SEED=20260310

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/raw_task_success_expanded_2026-03-10}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-results/raw_task_success_nonorm_2026-03-10}"
EPISODES="${EPISODES:-3}"
SEED="${SEED:-20260310}"
TRAIN_N=10
EVAL_N=30
SCENARIO_SET="expanded_n40"

read -r -a MODEL_KEYS <<< "${MODELS:-or_gpt4o_mini or_kimi_k2 or_qwen3_32b or_gpt4o or_deepseek_v3}"

run_condition() {
  local model="$1"
  local condition="$2"
  local output="$3"
  local extra_args="${4:-}"

  local args=(
    --config experiment.yaml
    --scenario-set "$SCENARIO_SET"
    --model "$model"
    --episodes "$EPISODES"
    --train-max-scenarios "$TRAIN_N"
    --eval-max-scenarios "$EVAL_N"
    --seed "$SEED"
    --output "$output"
  )

  if [[ "$condition" == "static" ]]; then
    args+=(--mode static)
  elif [[ "$condition" == "replay" ]]; then
    args+=(--mode learning --learning-strategy replay_blind)
  elif [[ "$condition" == "rules" ]]; then
    args+=(--mode learning --learning-strategy rules)
  fi

  if [[ -n "$extra_args" ]]; then
    args+=($extra_args)
  fi

  echo "[$(date -u +%H:%M:%S)] Running $model / $condition -> $output"
  infisical run -- uv run python src/run_task_success_evaluation.py "${args[@]}"
}

# --- Main conditions ---
for model in "${MODEL_KEYS[@]}"; do
  run_condition "$model" "static"  "$OUTPUT_ROOT/static_run"
  run_condition "$model" "replay"  "$OUTPUT_ROOT/replay_run"
  run_condition "$model" "rules"   "$OUTPUT_ROOT/rules_run"
done

# --- Normalization ablation (static only, all models) ---
for model in "${MODEL_KEYS[@]}"; do
  run_condition "$model" "static" "$ABLATION_OUTPUT_ROOT/static_nonorm_run" "--no-normalize"
done

echo "Done. Main results -> $OUTPUT_ROOT"
echo "Ablation results  -> $ABLATION_OUTPUT_ROOT"
