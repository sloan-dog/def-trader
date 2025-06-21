#!/bin/bash

# Terraform Minimal Setup for Local Development
# Uses tfrun.sh wrapper for proper authentication handling

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

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Create minimal GCP resources for local development

OPTIONS:
    --check     Check current resources without applying changes
    --help      Show this help message

EXAMPLES:
    $0              # Create minimal resources
    $0 --check      # Check what exists and what would be created
EOF
}

# Parse arguments
CHECK_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
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

# Change to terraform directory
SCRIPT_DIR="$(realpath $(dirname "$0"))"
TFRUN="$SCRIPT_DIR/tfrun.sh"
cd "$SCRIPT_DIR/../terraform"

# DEBUG
echo "DEBUGGING SCRIPT_DIR: ${SCRIPT_DIR}"
echo "DEBUGGING TFRUN DIR : ${TFRUN}"
# Check if tfrun.sh exists
if [ ! -x "$TFRUN" ]; then
    print_error "tfrun.sh not found or not executable!"
    print_info "Make sure tfrun.sh exists and is executable:"
    print_info "  chmod +x $TFRUN"
    exit 1
fi

print_info "🚀 Starting minimal Terraform setup for local development"
echo

# Initialize if needed
if [ ! -d ".terraform" ]; then
    print_info "Initializing Terraform..."
    $TFRUN init -backend-config=backend.conf
else
    print_info "Terraform already initialized"
fi

# Show current state
print_info "Current Terraform state:"
$TFRUN state list 2>/dev/null || echo "  (empty state)"
echo

# If check-only mode, show what would be created and exit
if [ "$CHECK_ONLY" = true ]; then
    print_info "Resources that would be created with full apply:"

    # Run a plan to see what would be created
    $TFRUN plan -var-file=terraform.tfvars -detailed-exitcode > /tmp/tfplan-output.txt 2>&1
    PLAN_EXIT_CODE=$?

    if [ $PLAN_EXIT_CODE -eq 0 ]; then
        print_success "Infrastructure is up to date - no changes needed"
    elif [ $PLAN_EXIT_CODE -eq 2 ]; then
        # Extract and show resources to be created
        grep "will be created" /tmp/tfplan-output.txt | sed 's/.*# /  - /' | sed 's/ will be created//' || true
        TO_CREATE=$(grep -c "will be created" /tmp/tfplan-output.txt || echo "0")
        echo
        print_info "Total resources to create: $TO_CREATE"
    else
        print_error "Error running terraform plan"
    fi

    rm -f /tmp/tfplan-output.txt
    exit 0
fi

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

# Run plan
print_info "Running terraform plan..."
if $TFRUN plan -var-file=terraform.tfvars $TARGET_ARGS -out=tfplan-minimal; then
    print_success "Plan created successfully"
else
    print_error "Terraform plan failed"
    exit 1
fi

echo
read -p "Do you want to apply these changes? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    print_warning "Terraform apply cancelled"
    rm -f tfplan-minimal
    exit 0
fi

# Apply the plan
print_info "Applying terraform plan..."
if $TFRUN apply tfplan-minimal; then
    print_success "Terraform apply completed successfully!"
    rm -f tfplan-minimal
else
    print_error "Terraform apply failed"
    rm -f tfplan-minimal
    exit 1
fi

echo
print_info "📊 Resources created:"
$TFRUN state list | grep -E "(bigquery|service_account|secret)" || true

echo
print_success "✅ Minimal infrastructure created!"
echo

# Get service account email
SERVICE_ACCOUNT_EMAIL=$($TFRUN output -raw service_account_email 2>/dev/null || echo "")

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
    echo "   python ../scripts/test_local_setup.py"
    echo
    echo "5. Run data ingestion:"
    echo "   python -m src.jobs.backfill_job --start-date 2024-01-01 --end-date 2024-01-31"
fi

print_info "💡 When ready to deploy full infrastructure, simply run:"
echo "   $TFRUN apply -var-file=terraform.tfvars"
echo
echo "This will add the remaining resources (Cloud Run, Scheduler, etc.) without conflicts!"