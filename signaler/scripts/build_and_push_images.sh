#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${PROJECT_ID:-trading-signals-420-69}"
REGISTRY="gcr.io/${PROJECT_ID}/trading-system"
DOCKER_BUILDKIT=1
export DOCKER_BUILDKIT

echo -e "${GREEN}Starting optimized Docker build process...${NC}"

# Configure Docker for GCR
echo -e "${YELLOW}Configuring Docker for Google Container Registry...${NC}"
gcloud auth configure-docker

# Build base image first (contains shared dependencies)
echo -e "${YELLOW}Building base image with core dependencies...${NC}"
docker build \
  --cache-from ${REGISTRY}/base:latest \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -f docker/Dockerfile.base \
  -t ${REGISTRY}/base:latest \
  -t trading-signal-base:latest \
  .

# Push base image
echo -e "${YELLOW}Pushing base image...${NC}"
docker push ${REGISTRY}/base:latest

# Build service images in parallel using the base image
echo -e "${YELLOW}Building service-specific images in parallel...${NC}"

# Function to build and push an image
build_and_push() {
  local service=$1
  local dockerfile=$2
  local tag=$3
  
  echo -e "${YELLOW}Building ${service} image...${NC}"
  docker build \
    --cache-from ${REGISTRY}/${tag}:latest \
    --build-arg BASE_IMAGE=${REGISTRY}/base:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -f ${dockerfile} \
    -t ${REGISTRY}/${tag}:latest \
    .
  
  echo -e "${YELLOW}Pushing ${service} image...${NC}"
  docker push ${REGISTRY}/${tag}:latest
  
  echo -e "${GREEN}✓ ${service} image built and pushed successfully${NC}"
}

# Build services in parallel
(build_and_push "API" "docker/Dockerfile.api" "prediction-service") &
PID1=$!

(build_and_push "Ingestion" "docker/Dockerfile.ingestion" "daily-ingestion") &
PID2=$!

# Only build training image if requested (it's the largest)
if [[ "$1" == "--with-training" ]]; then
  (build_and_push "Training" "docker/Dockerfile.training" "training") &
  PID3=$!
  wait $PID1 $PID2 $PID3
else
  wait $PID1 $PID2
  echo -e "${YELLOW}Skipping training image (use --with-training to include)${NC}"
fi

# Display image sizes
echo -e "${GREEN}Build complete! Image sizes:${NC}"
docker images | grep ${REGISTRY}

# Optional: Clean up dangling images
if [[ "$2" == "--cleanup" ]]; then
  echo -e "${YELLOW}Cleaning up dangling images...${NC}"
  docker image prune -f
fi

echo -e "${GREEN}All images built and pushed successfully!${NC}"
echo -e "${YELLOW}To deploy, run: ./scripts/terraform_full_deploy.sh --project-id ${PROJECT_ID}${NC}"