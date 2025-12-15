#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python "$SCRIPT_DIR/amazon18_data_process.py" \
    --dataset Industrial_and_Scientific \
    --user_k 5 \
    --item_k 5 \
    --st_year 2017 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 11 \
    --output_path "$SCRIPT_DIR/Amazon18" \
    "$@"
