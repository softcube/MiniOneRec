#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

DEFAULT_DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy}"
DEFAULT_CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/output/Industrial_and_Scientific}"
DEFAULT_LR="${LR:-1e-3}"
DEFAULT_EPOCHS="${EPOCHS:-10000}"
DEFAULT_BATCH_SIZE="${BATCH_SIZE:-20480}"

cmd=(
  python "$SCRIPT_DIR/rqvae.py"
  --data_path "$DEFAULT_DATA_PATH"
  --ckpt_dir "$DEFAULT_CKPT_DIR"
  --lr "$DEFAULT_LR"
  --epochs "$DEFAULT_EPOCHS"
  --batch_size "$DEFAULT_BATCH_SIZE"
)

cmd+=( "$@" )

printf '%s\n' "Running rqvae with:" \
  "  data_path:   $DEFAULT_DATA_PATH" \
  "  ckpt_dir:    $DEFAULT_CKPT_DIR" \
  "  lr:          $DEFAULT_LR" \
  "  epochs:      $DEFAULT_EPOCHS" \
  "  batch_size:  $DEFAULT_BATCH_SIZE" \
  "  extra args:  $*" \
  "Command:      ${cmd[*]}"

exec "${cmd[@]}"
