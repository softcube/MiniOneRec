#!/bin/bash
set -euo pipefail

# Set variables (same structure/sequence as `RQ-VAE-Recommender/run_container_aws.sh`)
REPOSITORY_NAME="minionerec"
IMAGE_TAG="aws"
MODEL_EXAMPLE_DIR="${MODEL_EXAMPLE_DIR:-/var/lib/models_data/minionerec/data}"
CONTAINER_MODEL_EXAMPLE_DIR="/root/minionerec/data"
OUTPUT_DIR="${OUTPUT_DIR:-/var/lib/models_data/minionerec/output}"
CONTAINER_OUTPUT_DIR="/root/minionerec/output"
CONTAINER_NAME="minionerec"
GPU_OPTION="${GPU_OPTION:-}"

# Defaults for Merlin embeddings (you can override via env vars).
DATA_PATH="${DATA_PATH:-./data/merlin/embeddings_matrix.npy}"
CKPT_DIR="${CKPT_DIR:-./output/merlin}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-20480}"

# Parse command line arguments
RUN_MODE="full"
if [ "${1:-}" = "container-only" ]; then
  RUN_MODE="container-only"
  shift
fi

# Step 1: Build the Docker image using build_image_aws.sh
echo "Building Docker image using build_image_aws.sh"
bash ./build_image_aws.sh

if ! docker image inspect "${REPOSITORY_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
  echo "ERROR: Docker image ${REPOSITORY_NAME}:${IMAGE_TAG} was not built successfully."
  exit 1
fi

# Check if container already exists
if [ "$(docker ps -a -q -f name=${CONTAINER_NAME})" ]; then
  echo "Container ${CONTAINER_NAME} already exists. Removing it..."
  docker stop ${CONTAINER_NAME} 2>/dev/null
  docker rm ${CONTAINER_NAME}
fi

# Step 2: Run the container in background mode with volume mounted
echo "Running Docker container in background mode with model data volume"
CONTAINER_ID="$(docker run -d \
  --name ${CONTAINER_NAME} \
  --restart always \
  ${GPU_OPTION} \
  -v ${MODEL_EXAMPLE_DIR}:${CONTAINER_MODEL_EXAMPLE_DIR} \
  -v ${OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR} \
  ${REPOSITORY_NAME}:${IMAGE_TAG})"

echo "Container started successfully: ${CONTAINER_ID}"

# If in container-only mode, exit here
if [ "$RUN_MODE" = "container-only" ]; then
  echo "Running in container-only mode. Container is ready but script will not execute."
  echo "Command to execute script manually:"
  echo "docker exec ${CONTAINER_NAME} bash rq/rqvae.sh --data_path ${DATA_PATH} --ckpt_dir ${CKPT_DIR} --lr ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE}"
  exit 0
fi

# Wait a moment for container to start
sleep 2

# Step 3: Execute the rqvae training inside the container
echo "Executing rq/rqvae.sh inside container"
if ! docker ps -q -f name="^${CONTAINER_NAME}$" | grep -q . ; then
  echo "ERROR: Container ${CONTAINER_NAME} is not running."
  docker ps -a --filter name="^${CONTAINER_NAME}$" || true
  exit 1
fi

DOCKER_EXEC_FLAGS=(-i)
if [ -t 1 ]; then
  DOCKER_EXEC_FLAGS=(-it)
fi

docker exec "${DOCKER_EXEC_FLAGS[@]}" ${CONTAINER_NAME} \
  bash rq/rqvae.sh \
    --data_path "${DATA_PATH}" \
    --ckpt_dir "${CKPT_DIR}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    "$@"
