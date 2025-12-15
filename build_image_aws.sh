#!/bin/bash
set -euo pipefail

# Set variables (same structure/sequence as `RQ-VAE-Recommender/build_image_aws.sh`)
REPOSITORY_NAME="${REPOSITORY_NAME:-minionerec}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
DOCKERFILE="${DOCKERFILE:-Dockerfile_aws}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04}"
INSTALL_FULL_REQUIREMENTS="${INSTALL_FULL_REQUIREMENTS:-1}"

# Optional compatibility knobs (not required by this repo, but kept to match baseline interface).
SSH_KEY="${SSH_KEY:-}"
GRAYLOG_ADDRESS="${GRAYLOG_ADDRESS:-}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Builds the Docker image for running MiniOneRec inside a container.

Usage:
  ./build_image_aws.sh

Env vars:
  REPOSITORY_NAME=minionerec
  IMAGE_TAG=latest
  DOCKERFILE=Dockerfile_aws
  PYTHON_VERSION=3.9
  BASE_IMAGE=nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04
  INSTALL_FULL_REQUIREMENTS=1   (set to 0 for minimal deps)
EOF
  exit 0
fi

# Build the Docker image using BuildKit (same pattern as baseline).
echo "Building Docker image with BuildKit"
DOCKER_BUILDKIT=1 docker build --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t ${REPOSITORY_NAME}:${IMAGE_TAG} \
  -f ${DOCKERFILE} \
  --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
  --build-arg BASE_IMAGE=${BASE_IMAGE} \
  --build-arg INSTALL_FULL_REQUIREMENTS=${INSTALL_FULL_REQUIREMENTS} \
  --build-arg SSH_KEY="${SSH_KEY}" \
  --build-arg GRAYLOG_ADDRESS="${GRAYLOG_ADDRESS}" \
  .

echo "Built image: ${REPOSITORY_NAME}:${IMAGE_TAG}"
