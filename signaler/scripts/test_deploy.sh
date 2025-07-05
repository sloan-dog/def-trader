set -o allexport
source .env
set +o allexport

IMAGE_NAME=gcr.io/${GCP_PROJECT_ID}/trading-system/daily-ingestion:latest

gcloud run deploy daily-ingestion \
  --image $IMAGE_NAME \
  --platform managed \
  --region $GCP_REGION \
  --no-allow-unauthenticated \
  --service-account trading-system@${GCP_PROJECT_ID}.iam.gserviceaccount.com