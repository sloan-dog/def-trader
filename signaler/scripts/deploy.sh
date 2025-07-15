#!/bin/bash

# GCP Deployment Script for Trading Signal System (pyproject.toml version)

set -e  # Exit on any error

# Default values
PROJECT_ID=""
REGION="us-central1"
PYTHON_VERSION="3.11.6"
BUILD_IMAGES=true
DEPLOY_TERRAFORM=true
SKIP_TESTS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Trading Signal System to Google Cloud Platform

OPTIONS:
    --project-id PROJECT_ID     GCP Project ID (required)
    --region REGION            GCP Region (default: us-central1)
    --python-version VERSION   Python version (default: 3.11.6)
    --skip-images              Skip building Docker images
    --skip-terraform           Skip Terraform deployment
    --skip-tests               Skip running tests before deployment
    --help                     Show this help message

EXAMPLES:
    $0 --project-id my-trading-project
    $0 --project-id my-project --region us-west1 --skip-tests
    $0 --project-id my-project --skip-images --skip-terraform
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --skip-images)
            BUILD_IMAGES=false
            shift
            ;;
        --skip-terraform)
            DEPLOY_TERRAFORM=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$PROJECT_ID" ]]; then
    print_error "Project ID is required. Use --project-id PROJECT_ID"
    show_usage
    exit 1
fi

print_status "Starting deployment to GCP..."
print_status "Project ID: $PROJECT_ID"
print_status "Region: $REGION"
print_status "Python Version: $PYTHON_VERSION"

# Check prerequisites
print_status "Checking prerequisites..."

# Check if required tools are installed
for tool in gcloud docker python terraform; do
    if ! command -v $tool &> /dev/null; then
        print_error "$tool is not installed or not in PATH"
        exit 1
    fi
done

# Check Python version
current_python_version=$(python --version 2>&1 | cut -d' ' -f2)
if [[ "$current_python_version" != "$PYTHON_VERSION"* ]]; then
    print_warning "Current Python version ($current_python_version) doesn't match target ($PYTHON_VERSION)"
    if command -v pyenv &> /dev/null; then
        print_status "Using pyenv to switch to Python $PYTHON_VERSION"
        pyenv local $PYTHON_VERSION
    fi
fi

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "Not authenticated with gcloud. Run: gcloud auth login"
    exit 1
fi

# Set the project
print_status "Setting GCP project to $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Check if project exists and is accessible
if ! gcloud projects describe $PROJECT_ID &> /dev/null; then
    print_error "Cannot access project $PROJECT_ID. Check project ID and permissions."
    exit 1
fi

# Enable required APIs
print_status "Enabling required GCP APIs..."
apis=(
    "bigquery.googleapis.com"
    "run.googleapis.com"
    "cloudscheduler.googleapis.com"
    "cloudbuild.googleapis.com"
    "containerregistry.googleapis.com"
    "aiplatform.googleapis.com"
    "storage.googleapis.com"
    "secretmanager.googleapis.com"
)

for api in "${apis[@]}"; do
    print_status "Enabling $api..."
    gcloud services enable $api
done

# Run tests before deployment (unless skipped)
if [[ "$SKIP_TESTS" == false ]]; then
    print_status "Running tests before deployment..."
    if [[ -f "pyproject.toml" ]]; then
        pip install -e ".[test,lint]"
        pytest tests/ -v -m "not integration" || {
            print_error "Tests failed. Use --skip-tests to deploy anyway."
            exit 1
        }
        print_status "All tests passed!"
    else
        print_warning "No pyproject.toml found, skipping tests"
    fi
fi

# Build and push Docker images
if [[ "$BUILD_IMAGES" == true ]]; then
    print_status "Building and pushing Docker images..."

    # Configure Docker for GCR
    gcloud auth configure-docker

    # Build images
    images=("base" "ingestion" "api")
    for image in "${images[@]}"; do
        print_status "Building $image image..."
        docker build \
            -f docker/Dockerfile.$image \
            -t gcr.io/$PROJECT_ID/trading-system/$image:latest \
            -t gcr.io/$PROJECT_ID/trading-system/$image:$(git rev-parse --short HEAD) \
            --build-arg PYTHON_VERSION=$PYTHON_VERSION \
            .

        print_status "Pushing $image image..."
        docker push gcr.io/$PROJECT_ID/trading-system/$image:latest
        docker push gcr.io/$PROJECT_ID/trading-system/$image:$(git rev-parse --short HEAD)
    done
fi

# Deploy infrastructure with Terraform
if [[ "$DEPLOY_TERRAFORM" == true ]]; then
    print_status "Deploying infrastructure with Terraform..."

    cd terraform

    # Initialize Terraform
    terraform init

    # Plan deployment
    terraform plan \
        -var="project_id=$PROJECT_ID" \
        -var="region=$REGION" \
        -out=tfplan

    # Apply deployment
    print_status "Applying Terraform configuration..."
    terraform apply tfplan

    cd ..

    print_status "Infrastructure deployment completed!"
fi

# Deploy Cloud Run services
print_status "Deploying Cloud Run services..."

# Deploy daily ingestion service
print_status "Deploying daily ingestion service..."
gcloud run deploy daily-ingestion \
    --image gcr.io/$PROJECT_ID/trading-system/ingestion:latest \
    --platform managed \
    --region $REGION \
    --no-allow-unauthenticated \
    --service-account trading-system@$PROJECT_ID.iam.gserviceaccount.com \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10

# Deploy API service
print_status "Deploying API service..."
gcloud run deploy prediction-api \
    --image gcr.io/$PROJECT_ID/trading-system/api:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --service-account trading-system@$PROJECT_ID.iam.gserviceaccount.com \
    --memory 8Gi \
    --cpu 4 \
    --timeout 900 \
    --max-instances 10

# Deploy backfill service
print_status "Deploying backfill service..."
gcloud run deploy backfill-service \
    --image gcr.io/$PROJECT_ID/trading-system/api:latest \
    --platform managed \
    --region $REGION \
    --no-allow-unauthenticated \
    --service-account trading-system@$PROJECT_ID.iam.gserviceaccount.com \
    --memory 8Gi \
    --cpu 4 \
    --timeout 3600 \
    --max-instances 5 \
    --command uvicorn,src.api.backfill_service:app,--host,0.0.0.0,--port,8080

# Get service URLs
INGESTION_URL=$(gcloud run services describe daily-ingestion --region=$REGION --format="value(status.url)")
API_URL=$(gcloud run services describe prediction-api --region=$REGION --format="value(status.url)")
BACKFILL_URL=$(gcloud run services describe backfill-service --region=$REGION --format="value(status.url)")

print_status "Service URLs:"
print_status "  Ingestion Service: $INGESTION_URL"
print_status "  API Service: $API_URL"
print_status "  Backfill Service: $BACKFILL_URL"

# Update Cloud Scheduler jobs
print_status "Setting up Cloud Scheduler jobs..."

# Create daily ingestion job
gcloud scheduler jobs create http daily-data-ingestion \
    --location=$REGION \
    --schedule="0 18 * * MON-FRI" \
    --uri="$INGESTION_URL/run" \
    --http-method=POST \
    --oidc-service-account-email=trading-system@$PROJECT_ID.iam.gserviceaccount.com \
    --time-zone="America/New_York" \
    --max-retry-attempts=3 \
    --max-retry-duration=600s \
    --min-backoff-duration=5s \
    --max-backoff-duration=300s \
    --attempt-deadline=3600s || {
        print_warning "Scheduler job might already exist, updating..."
        gcloud scheduler jobs update http daily-data-ingestion \
            --location=$REGION \
            --schedule="0 18 * * MON-FRI" \
            --uri="$INGESTION_URL/run"
    }

# Test API health
print_status "Testing API health..."
if curl -f "$API_URL/health" &> /dev/null; then
    print_status "API health check passed!"
else
    print_warning "API health check failed. Service might still be starting up."
fi

# Deployment summary
print_status "Deployment completed successfully! 🎉"
print_status ""
print_status "Summary:"
print_status "  Project: $PROJECT_ID"
print_status "  Region: $REGION"
print_status "  API URL: $API_URL"
print_status "  Ingestion URL: $INGESTION_URL"
print_status ""
print_status "Next steps:"
print_status "  1. Set up your Alpha Vantage API key in Secret Manager"
print_status "  2. Run initial data backfill"
print_status "  3. Train your first model"
print_status "  4. Set up monitoring and alerts"
print_status ""
print_status "Commands to get started:"
print_status "  # Set API key"
print_status "  gcloud secrets versions add alpha-vantage-api-key --data-file=<(echo 'YOUR_API_KEY')"
print_status ""
print_status "  # Run backfill"
print_status "  curl -X POST '$INGESTION_URL/backfill' -H 'Authorization: Bearer \$(gcloud auth print-identity-token)'"
print_status ""
print_status "  # Check API"
print_status "  curl '$API_URL/health'"