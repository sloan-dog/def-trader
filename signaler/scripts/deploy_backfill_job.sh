#!/bin/bash

# Deploy Cloud Run Job for backfill
# Usage: ./scripts/deploy_backfill_job.sh [project_id] [region]

set -e

# Default values
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
JOB_NAME="backfill-job"
IMAGE_NAME="gcr.io/${PROJECT_ID}/backfill-job:latest"

echo "🚀 Deploying Cloud Run Job: ${JOB_NAME}"
echo "📦 Project: ${PROJECT_ID}"
echo "🌍 Region: ${REGION}"
echo "🐳 Image: ${IMAGE_NAME}"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build \
    -f docker/Dockerfile.backfill.job \
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
    --source=cloud-run-backfill-job.yaml

echo "✅ Deployment complete!"
echo "📊 Monitor job execution:"
echo "   gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION}"
echo ""
echo "🔍 View logs:"
echo "   gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}\" --project=${PROJECT_ID} --limit=50"
echo ""
echo "🚀 Execute jobs:"
echo "   # Hourly backfill"
echo "   gcloud run jobs execute ${JOB_NAME} --region=${REGION} --args=\"--type=hourly\""
echo ""
echo "   # Daily backfill"
echo "   gcloud run jobs execute ${JOB_NAME} --region=${REGION} --args=\"--type=daily\""
echo ""
echo "   # Weekly backfill"
echo "   gcloud run jobs execute ${JOB_NAME} --region=${REGION} --args=\"--type=weekly\""
echo ""
echo "   # Historical backfill"
echo "   gcloud run jobs execute ${JOB_NAME} --region=${REGION} --args=\"--type=historical,--start-year=2020,--end-year=2024,--data-types=ohlcv,--data-types=macro\"" 