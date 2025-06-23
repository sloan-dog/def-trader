#!/bin/bash

# Full Terraform Deployment Script for CI/CD
# This deploys ALL infrastructure resources for the trading system

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
AUTO_APPROVE=false
DESTROY_MODE=false

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

Deploy the complete Trading Signal System infrastructure to GCP

OPTIONS:
    --project-id PROJECT_ID    GCP Project ID (required unless set in env)
    --region REGION           GCP Region (default: us-central1)
    --environment ENV         Environment (dev/staging/prod, default: prod)
    --auto-approve           Skip confirmation prompts
    --destroy                Destroy all infrastructure
    --plan-only              Only show plan, don't apply
    --help                   Show this help message

ENVIRONMENT VARIABLES:
    GCP_PROJECT_ID           Can be set instead of --project-id
    ALPHA_VANTAGE_API_KEY    Required for deployment
    ALERT_EMAIL             Email for alerts (optional)

EXAMPLES:
    $0 --project-id my-project --auto-approve
    $0 --project-id my-project --plan-only
    $0 --project-id my-project --environment staging
    $0 --destroy --auto-approve

PREREQUISITES:
    - Service account key at ~/trading-system-key.json
    - terraform/terraform.tfvars file configured
    - tfrun.sh script available
EOF
}

# Parse command line arguments
PLAN_ONLY=false
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
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --auto-approve)
            AUTO_APPROVE=true
            shift
            ;;
        --destroy)
            DESTROY_MODE=true
            shift
            ;;
        --plan-only)
            PLAN_ONLY=true
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
    print_error "Project ID is required. Use --project-id or set GCP_PROJECT_ID"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFRUN="$SCRIPT_DIR/tfrun.sh"

# Check if tfrun.sh exists
if [[ ! -x "$TFRUN" ]]; then
    print_error "tfrun.sh not found or not executable at: $TFRUN"
    print_info "Make sure tfrun.sh exists and is executable:"
    print_info "  chmod +x $TFRUN"
    exit 1
fi

# Header
echo
echo -e "${BLUE}🚀 Trading Signal System - Full Infrastructure Deployment${NC}"
echo "========================================================="
echo "Project:     $PROJECT_ID"
echo "Region:      $REGION"
echo "Environment: $ENVIRONMENT"
echo "Mode:        $([ "$DESTROY_MODE" = true ] && echo "DESTROY" || echo "DEPLOY")"
echo

# Export project ID for tfrun.sh
export GCP_PROJECT_ID="$PROJECT_ID"

VAR_FILE_PATH="../terraform/terraform.tfvars"

echo "WORKING DIR: $(pwd)"
# Check if terraform.tfvars exists
if [[ ! -f "$VAR_FILE_PATH" ]]; then
    print_error "terraform/terraform.tfvars not found!"
    print_info "Creating template terraform.tfvars..."

    # Get user email
    USER_EMAIL=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || echo "admin@example.com")

    cat > "$VAR_FILE_PATH" << EOF
project_id = "$PROJECT_ID"
region = "$REGION"
bigquery_dataset = "trading_signals"
alpha_vantage_api_key = "${ALPHA_VANTAGE_API_KEY:-YOUR_API_KEY_HERE}"
alert_email = "${ALERT_EMAIL:-$USER_EMAIL}"
environment = "$ENVIRONMENT"
EOF

    print_warning "Please edit terraform/terraform.tfvars with your actual values before proceeding"
    exit 1
fi

CONF_PATH="../terraform/backend.conf"
# Initialize Terraform using tfrun.sh
print_info "Initializing Terraform..."
if [[ -f "$CONF_PATH" ]]; then
    $TFRUN init -backend-config=backend.conf -upgrade
else
    # Create backend config if it doesn't exist
    cat > "$CONF_PATH" << EOF
bucket = "${PROJECT_ID}-terraform-state"
prefix = "terraform/state"
EOF

    # Create the state bucket
    print_info "Creating Terraform state bucket..."
    gsutil mb -p "$PROJECT_ID" "gs://${PROJECT_ID}-terraform-state" 2>/dev/null || true
    gsutil versioning set on "gs://${PROJECT_ID}-terraform-state"

    $TFRUN init -backend-config=backend.conf
fi

# Validate configuration
print_info "Validating Terraform configuration..."
if ! $TFRUN validate; then
    print_error "Terraform validation failed"
    exit 1
fi

# Show current state
print_info "Current infrastructure state:"
$TFRUN state list 2>/dev/null | head -20 || echo "  (empty state)"
RESOURCE_COUNT=$($TFRUN state list 2>/dev/null | wc -l || echo "0")
if [[ $RESOURCE_COUNT -gt 0 ]]; then
    print_info "Total resources in state: $RESOURCE_COUNT"
fi
echo

# Plan the deployment
if [[ "$DESTROY_MODE" = true ]]; then
    print_warning "Planning infrastructure DESTRUCTION..."
    $TFRUN plan -destroy -out=tfplan
else
    print_info "Planning infrastructure deployment..."
    $TFRUN plan -out=tfplan
fi

# Show plan summary
echo
print_info "Plan Summary:"
$TFRUN show -no-color tfplan | grep -E "will be created|will be destroyed|will be updated" | sort | uniq -c || true
echo

# If plan-only mode, exit here
if [[ "$PLAN_ONLY" = true ]]; then
    print_info "Plan complete. Use without --plan-only to apply changes."
    rm -f terraform/tfplan
    exit 0
fi

# Confirm before applying
if [[ "$AUTO_APPROVE" != true ]]; then
    echo
    if [[ "$DESTROY_MODE" = true ]]; then
        print_warning "⚠️  WARNING: This will DESTROY all infrastructure! ⚠️"
        read -p "Are you SURE you want to destroy everything? Type 'destroy' to confirm: " confirm
        if [[ "$confirm" != "destroy" ]]; then
            print_info "Destruction cancelled"
            rm -f terraform/tfplan
            exit 0
        fi
    else
        read -p "Do you want to apply these changes? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            print_info "Deployment cancelled"
            rm -f terraform/tfplan
            exit 0
        fi
    fi
fi

# Apply the changes
if [[ "$DESTROY_MODE" = true ]]; then
    print_warning "Destroying infrastructure..."
else
    print_info "Applying infrastructure changes..."
fi

if $TFRUN apply tfplan; then
    print_success "Terraform apply completed successfully!"
    rm -f terraform/tfplan
else
    print_error "Terraform apply failed"
    rm -f terraform/tfplan
    exit 1
fi

# Show outputs
if [[ "$DESTROY_MODE" != true ]]; then
    echo
    print_info "📊 Deployment Outputs:"
    $TFRUN output -json | jq -r 'to_entries[] | "  \(.key): \(.value.value)"' 2>/dev/null || true

    echo
    print_info "🔗 Important URLs:"

    # Get Cloud Run URLs if available
    INGESTION_URL=$($TFRUN output -raw daily_ingestion_url 2>/dev/null || echo "")
    API_URL=$($TFRUN output -raw prediction_api_url 2>/dev/null || echo "")

    if [[ -n "$INGESTION_URL" ]]; then
        echo "  Daily Ingestion: $INGESTION_URL"
    fi
    if [[ -n "$API_URL" ]]; then
        echo "  Prediction API: $API_URL"
    fi

    echo
    print_success "✅ Infrastructure deployment complete!"
    echo
    print_info "📋 Next Steps:"
    echo "  1. Set Alpha Vantage API key in Secret Manager:"
    echo "     echo -n 'YOUR_API_KEY' | gcloud secrets versions add alpha-vantage-api-key --data-file=-"
    echo
    echo "  2. Test the deployment:"
    echo "     curl $API_URL/health"
    echo
    echo "  3. Run initial data backfill:"
    echo "     gcloud run jobs create backfill-job \\"
    echo "       --image gcr.io/$PROJECT_ID/trading-system/ingestion:latest \\"
    echo "       --region $REGION \\"
    echo "       --memory 4Gi --cpu 2 --task-timeout 3600 \\"
    echo "       --service-account trading-system@$PROJECT_ID.iam.gserviceaccount.com \\"
    echo "       --command trading-backfill"
    echo
    echo "  4. Monitor services:"
    echo "     gcloud run services list --region $REGION"
    echo "     gcloud scheduler jobs list --location $REGION"

    # Create a deployment summary file
    cat > deployment-summary.txt << EOF
Trading Signal System Deployment Summary
======================================
Date: $(date)
Project: $PROJECT_ID
Region: $REGION
Environment: $ENVIRONMENT

Resources Created:
$($TFRUN state list | wc -l) total resources

Key Components:
- BigQuery Dataset: trading_signals
- Cloud Run Services: daily-ingestion, prediction-api
- Cloud Scheduler Jobs: daily-data-ingestion
- Service Account: trading-system@$PROJECT_ID.iam.gserviceaccount.com

URLs:
- Ingestion Service: $INGESTION_URL
- Prediction API: $API_URL

Next Steps:
1. Configure Alpha Vantage API key in Secret Manager
2. Run initial data backfill
3. Train first model
4. Enable monitoring
EOF

    print_info "Deployment summary saved to: deployment-summary.txt"
else
    print_success "✅ Infrastructure destroyed successfully!"
fi

echo
print_info "Run '$TFRUN state list' to see current resources"