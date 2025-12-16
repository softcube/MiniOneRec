#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python "$SCRIPT_DIR/merlin_data_process.py" \
    --dataset merlin \
    --output_path "$SCRIPT_DIR/merlin" \
    --train_file "$SCRIPT_DIR/raw/merlin/train.parquet" \
    --valid_file "$SCRIPT_DIR/raw/merlin/valid.parquet" \
    --test_file "$SCRIPT_DIR/raw/merlin/test.parquet" \
    "$@"

