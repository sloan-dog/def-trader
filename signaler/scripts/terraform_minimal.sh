#!/bin/bash

# Terraform Minimal Setup for Local Development
# This script creates only the essential GCP resources needed for local development
# Later, you can run full tofu apply without any conflicts

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Change to terraform directory
cd "$(dirname "$0")/../terraform"

print_info "🚀 Starting minimal Terraform setup for local development"
echo

# Check if terraform is initialized
if [ ! -d ".terraform" ]; then
    print_info "Initializing Terraform..."
    tofu init -backend-config=backend.conf
else
    print_info "Terraform already initialized"
fi

# Show current state
print_info "Current Terraform state:"
terraform state list 2>/dev/null || echo "  (empty state)"
echo

# Define the minimal resources needed for local development
TARGETS=(
    # Enable required APIs
    "google_project_service.required_apis[\"bigquery.googleapis.com\"]"
    "google_project_service.required_apis[\"storage.googleapis.com\"]"
    "google_project_service.required_apis[\"secretmanager.googleapis.com\"]"

    # Service account for authentication
    "google_service_account.trading_system"

    # IAM roles for the service account
    "google_project_iam_member.service_account_roles[\"roles/bigquery.dataEditor\"]"
    "google_project_iam_member.service_account_roles[\"roles/bigquery.jobUser\"]"
    "google_project_iam_member.service_account_roles[\"roles/storage.objectAdmin\"]"
    "google_project_iam_member.service_account_roles[\"roles/secretmanager.secretAccessor\"]"

    # BigQuery dataset and all tables
    "module.bigquery"

    # Secret Manager for API keys
    "google_secret_manager_secret.alpha_vantage_key"
    "google_secret_manager_secret_version.alpha_vantage_key"

    # Model storage bucket (optional but useful)
    "google_storage_bucket.models"
)

print_info "Planning to create the following resources:"
echo
for target in "${TARGETS[@]}"; do
    echo "  • $target"
done
echo

# Build the -target arguments
TARGET_ARGS=""
for target in "${TARGETS[@]}"; do
    TARGET_ARGS="$TARGET_ARGS -target=$target"
done

# Run tofu plan
print_info "Running tofu plan..."
if tofu plan -var-file=terraform.tfvars $TARGET_ARGS -out=tfplan-minimal; then
    print_success "Plan created successfully"
else
    print_error "tofu plan failed"
    exit 1
fi

echo
read -p "Do you want to apply these changes? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    print_warning "tofu apply cancelled"
    rm -f tfplan-minimal
    exit 0
fi

# Apply the plan
print_info "Applying tofu plan..."
if tofu apply tfplan-minimal; then
    print_success "tofu apply completed successfully!"
    rm -f tfplan-minimal
else
    print_error "tofu apply failed"
    rm -f tfplan-minimal
    exit 1
fi

echo
print_info "📊 Resources created:"
terraform state list | grep -E "(bigquery|service_account|secret)" || true

echo
print_success "✅ Minimal infrastructure created!"
echo

# Get service account email
SERVICE_ACCOUNT_EMAIL=$(terraform output -raw service_account_email 2>/dev/null || echo "")

if [ -n "$SERVICE_ACCOUNT_EMAIL" ]; then
    print_info "📋 Next steps:"
    echo
    echo "1. Create a service account key for local development:"
    echo "   gcloud iam service-accounts keys create ~/trading-system-key.json \\"
    echo "     --iam-account=$SERVICE_ACCOUNT_EMAIL"
    echo
    echo "2. Set the environment variable:"
    echo "   export GOOGLE_APPLICATION_CREDENTIALS=~/trading-system-key.json"
    echo
    echo "3. Add to your .env file:"
    echo "   echo 'GOOGLE_APPLICATION_CREDENTIALS=\$HOME/trading-system-key.json' >> ../.env"
    echo
    echo "4. Test your setup:"
    echo "   python ../test_setup.py"
    echo
    echo "5. Run data ingestion:"
    echo "   python -m src.jobs.backfill_job --start-date 2024-01-01 --end-date 2024-01-31"
fi

print_info "💡 When ready to deploy full infrastructure, simply run:"
echo "   tofu apply -var-file=terraform.tfvars"
echo
echo "This will add the remaining resources (Cloud Run, Scheduler, etc.) without conflicts!"