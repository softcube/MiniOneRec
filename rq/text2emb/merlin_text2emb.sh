#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

DEFAULT_DATASET="${DATASET:-merlin}"
DEFAULT_ROOT="${ROOT:-$REPO_ROOT/data/merlin/$DEFAULT_DATASET}"

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

if ! has_flag --embedding_input_source "${USER_ARGS[@]}"; then
    if [[ -z "${EMBEDDING_INPUT_SOURCE:-}" ]]; then
        echo "Error: provide --embedding_input_source <path> or set the EMBEDDING_INPUT_SOURCE environment variable." >&2
        exit 1
    fi
    CMD_ARGS+=( --embedding_input_source "$EMBEDDING_INPUT_SOURCE" )
fi

python "$SCRIPT_DIR/merlin_remap_emb.py" \
    "${CMD_ARGS[@]}" \
    "${USER_ARGS[@]}"

