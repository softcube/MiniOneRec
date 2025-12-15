#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

DEFAULT_DATASET="${DATASET:-Industrial_and_Scientific}"
DEFAULT_ROOT="${ROOT:-$REPO_ROOT/data/Amazon18/$DEFAULT_DATASET}"
DEFAULT_PLM_NAME="${PLM_NAME:-qwen}"
DEFAULT_BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

USER_ARGS=( "$@" )

has_flag() {
    local flag="$1"
    shift
    for arg in "$@"; do
        if [[ "$arg" == "$flag" ]]; then
            return 0
        fi
    done
    return 1
}

CMD_ARGS=()

if ! has_flag --dataset "${USER_ARGS[@]}"; then
    CMD_ARGS+=( --dataset "$DEFAULT_DATASET" )
fi

if ! has_flag --root "${USER_ARGS[@]}"; then
    CMD_ARGS+=( --root "$DEFAULT_ROOT" )
fi

if ! has_flag --plm_name "${USER_ARGS[@]}"; then
    CMD_ARGS+=( --plm_name "$DEFAULT_PLM_NAME" )
fi

if ! has_flag --batch_size "${USER_ARGS[@]}"; then
    CMD_ARGS+=( --batch_size "$DEFAULT_BATCH_SIZE" )
fi

if ! has_flag --plm_checkpoint "${USER_ARGS[@]}"; then
    if [[ -z "${PLM_CHECKPOINT:-}" ]]; then
        echo "Error: provide --plm_checkpoint <path> or set the PLM_CHECKPOINT environment variable." >&2
        exit 1
    fi
    CMD_ARGS+=( --plm_checkpoint "$PLM_CHECKPOINT" )
fi

accelerate launch --num_processes "$NUM_PROCESSES" "$SCRIPT_DIR/amazon_text2emb.py" \
    "${CMD_ARGS[@]}" \
    "${USER_ARGS[@]}"
