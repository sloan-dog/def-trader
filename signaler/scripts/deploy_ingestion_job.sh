#!/bin/bash

# Deploy Cloud Run Job for daily ingestion
# Usage: ./scripts/deploy_ingestion_job.sh [project_id] [region]

set -e

# Default values
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
JOB_NAME="daily-ingestion-job"
IMAGE_NAME="gcr.io/${PROJECT_ID}/daily-ingestion-job:latest"

echo "🚀 Deploying Cloud Run Job: ${JOB_NAME}"
echo "📦 Project: ${PROJECT_ID}"
echo "🌍 Region: ${REGION}"
echo "🐳 Image: ${IMAGE_NAME}"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build \
    -f docker/Dockerfile.ingestion.job \
    -t ${IMAGE_NAME} \
    .

# Push to Container Registry
echo "📤 Pushing image to Container Registry..."
docker push ${IMAGE_NAME}

# Deploy Cloud Run Job
echo "🚀 Deploying Cloud Run Job..."
gcloud run jobs replace \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --source=cloud-run-job.yaml

echo "✅ Deployment complete!"
echo "📊 Monitor job execution:"
echo "   gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION}"
echo ""
echo "🔍 View logs:"
echo "   gcloud run jobs executions logs --job=${JOB_NAME} --region=${REGION}" 