#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-minionerec:aws}"
USE_GPU="${USE_GPU:-1}"
SHM_SIZE="${SHM_SIZE:-16g}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-minionerec-rqvae}"

# shellcheck disable=SC2016
usage() {
  cat <<'EOF'
Runs `rq/rqvae.sh` inside a Docker container.

Usage:
  IMAGE_NAME=minionerec:aws ./run_container_aws.sh [container-only] [extra rqvae.py args...]

Env vars (defaults match README):
  DATA_PATH=./data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy
  CKPT_DIR=./output/Industrial_and_Scientific
  LR=1e-3
  EPOCHS=10000
  BATCH_SIZE=20480
  USE_GPU=1
  SHM_SIZE=16g
  SKIP_BUILD=0
  CONTAINER_NAME=minionerec-rqvae

Examples:
  ./run_container_aws.sh
  SKIP_BUILD=1 ./run_container_aws.sh --device cuda:0
  ./run_container_aws.sh container-only
EOF
}

RUN_MODE="full"
if [[ "${1:-}" == "container-only" ]]; then
  RUN_MODE="container-only"
  shift
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Ensure the Docker image exists (build unless skipped explicitly).
if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "Building Docker image via build_image_aws.sh"
  bash "$REPO_ROOT/build_image_aws.sh"
else
  echo "SKIP_BUILD=1 → skipping build_image_aws.sh invocation"
fi

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "ERROR: Docker image $IMAGE_NAME not available. Run build_image_aws.sh first." >&2
  exit 1
fi

# Defaults mirror the README example.
DATA_PATH="${DATA_PATH:-./data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy}"
CKPT_DIR="${CKPT_DIR:-./output/Industrial_and_Scientific}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-20480}"

CONTAINER_DATA_PATH="$DATA_PATH"
extra_mounts=()
if [[ "$DATA_PATH" == /* ]]; then
  if [[ ! -f "$DATA_PATH" ]]; then
    echo "DATA_PATH does not exist: $DATA_PATH" >&2
    exit 1
  fi
  host_data_dir="$(cd "$(dirname "$DATA_PATH")" && pwd)"
  host_data_base="$(basename "$DATA_PATH")"
  extra_mounts+=( -v "$host_data_dir:/mnt/data:ro" )
  CONTAINER_DATA_PATH="/mnt/data/$host_data_base"
fi

CONTAINER_CKPT_DIR="$CKPT_DIR"
if [[ "$CKPT_DIR" == /* ]]; then
  mkdir -p "$CKPT_DIR"
  host_ckpt_dir="$(cd "$CKPT_DIR" && pwd)"
  extra_mounts+=( -v "$host_ckpt_dir:/mnt/output" )
  CONTAINER_CKPT_DIR="/mnt/output"
fi

docker_run_args=(
  --name "$CONTAINER_NAME"
  --restart always
  --ipc=host
  --shm-size "$SHM_SIZE"
  -e PYTHONUNBUFFERED=1
  -v "$REPO_ROOT:/workspace"
  -w /workspace
  --user "$(id -u):$(id -g)"
)

if [[ "$USE_GPU" == "1" ]]; then
  docker_run_args+=( --gpus all )
fi

docker_run_args+=( "${extra_mounts[@]}" )

if docker ps -a -q -f name="^${CONTAINER_NAME}$" >/dev/null 2>&1; then
  echo "Container ${CONTAINER_NAME} already exists. Removing it..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "Starting container ${CONTAINER_NAME} in background..."
docker run -d "${docker_run_args[@]}" "$IMAGE_NAME" tail -f /dev/null >/dev/null

cmd=(
  bash rq/rqvae.sh
  --data_path "$CONTAINER_DATA_PATH"
  --ckpt_dir "$CONTAINER_CKPT_DIR"
  --lr "$LR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
)
cmd+=( "$@" )

if [[ "$RUN_MODE" == "container-only" ]]; then
  echo "Container-only mode: container ${CONTAINER_NAME} is running."
  echo "Run the following manually when ready:"
  echo "docker exec -it ${CONTAINER_NAME} ${cmd[*]}"
  exit 0
fi

DOCKER_EXEC_FLAGS=(-i)
if [[ -t 1 ]]; then
  DOCKER_EXEC_FLAGS=(-it)
fi

echo "Executing training inside container ${CONTAINER_NAME}"
docker exec "${DOCKER_EXEC_FLAGS[@]}" -w /workspace "$CONTAINER_NAME" "${cmd[@]}"

#echo "Stopping container ${CONTAINER_NAME}"
#docker stop "${CONTAINER_NAME}" >/dev/null
#docker rm "${CONTAINER_NAME}" >/dev/null
