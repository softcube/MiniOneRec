#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="${IMAGE_NAME:-minionerec:aws}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04}"
INSTALL_FULL_REQUIREMENTS="${INSTALL_FULL_REQUIREMENTS:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Builds the Docker image for running MiniOneRec inside a container.

Usage:
  IMAGE_NAME=minionerec:aws ./build_image_aws.sh

Env vars:
  IMAGE_NAME=minionerec:aws
  BASE_IMAGE=nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04
  INSTALL_FULL_REQUIREMENTS=1   (set to 0 for minimal deps)
EOF
  exit 0
fi

docker build \
  -f "$REPO_ROOT/Dockerfile_aws" \
  -t "$IMAGE_NAME" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg INSTALL_FULL_REQUIREMENTS="$INSTALL_FULL_REQUIREMENTS" \
  "$REPO_ROOT"

echo "Built image: $IMAGE_NAME"
