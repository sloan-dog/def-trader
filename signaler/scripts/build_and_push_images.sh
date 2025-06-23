# Configure Docker for GCR
gcloud auth configure-docker

# Build base image
docker build -f docker/Dockerfile.base -t gcr.io/trading-signals-420-69/trading-system/base:latest .
docker push gcr.io/trading-signals-420-69/trading-system/base:latest

# Build ingestion image
docker build -f docker/Dockerfile.ingestion -t gcr.io/trading-signals-420-69/trading-system/daily-ingestion:latest .
docker push gcr.io/trading-signals-420-69/trading-system/daily-ingestion:latest

# Build API image (assuming you have one)
docker build -f docker/Dockerfile.api -t gcr.io/trading-signals-420-69/trading-system/prediction-service:latest .
docker push gcr.io/trading-signals-420-69/trading-system/prediction-service:latest

## Now run Terraform
#./scripts/terraform_full_deploy.sh --project-id trading-signals-420-69