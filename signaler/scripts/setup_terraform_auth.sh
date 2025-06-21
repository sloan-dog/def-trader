#!/bin/bash

# Setup Terraform Authentication Script
# This script configures the service account credentials for Terraform

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
PROJECT_ID="trading-signals-420-69"
SERVICE_ACCOUNT_EMAIL="trading-system@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="$HOME/trading-system-key.json"
ENV_FILE=".env"

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

# Header
echo -e "${BLUE}🔧 Terraform Authentication Setup${NC}"
echo "=================================="
echo

# Check if we're in the right directory
if [ ! -f "terraform/main.tf" ]; then
    print_error "This script should be run from the project root (where terraform/ directory exists)"
    exit 1
fi

# Step 1: Check if key file exists
if [ -f "$KEY_FILE" ]; then
    print_info "Service account key exists at: $KEY_FILE"

    # Verify it's valid JSON and for the right account
    if command -v jq &> /dev/null; then
        KEY_EMAIL=$(jq -r '.client_email' "$KEY_FILE" 2>/dev/null || echo "")
        if [ "$KEY_EMAIL" == "$SERVICE_ACCOUNT_EMAIL" ]; then
            print_success "Key file is valid for: $SERVICE_ACCOUNT_EMAIL"
        else
            print_error "Key file is for wrong account: $KEY_EMAIL"
            print_info "Expected: $SERVICE_ACCOUNT_EMAIL"
            exit 1
        fi
    fi
else
    print_warning "Service account key not found"
    echo
    read -p "Do you want to create a new service account key? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Creating service account key..."

        # Check if service account exists
        if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
            # Create the key
            gcloud iam service-accounts keys create "$KEY_FILE" \
                --iam-account="$SERVICE_ACCOUNT_EMAIL" \
                --project="$PROJECT_ID"

            # Set proper permissions
            chmod 600 "$KEY_FILE"
            print_success "Service account key created successfully"
        else
            print_error "Service account $SERVICE_ACCOUNT_EMAIL does not exist"
            print_info "Run the full Terraform setup first to create the service account"
            exit 1
        fi
    else
        print_error "Cannot proceed without service account key"
        exit 1
    fi
fi

# Step 2: Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"
print_info "Set GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE"

# Step 3: Update .env file
if [ -f "$ENV_FILE" ]; then
    # Remove old entry if exists
    grep -v "GOOGLE_APPLICATION_CREDENTIALS" "$ENV_FILE" > "$ENV_FILE.tmp" || true
    mv "$ENV_FILE.tmp" "$ENV_FILE"
fi

# Add new entry
echo "GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE" >> "$ENV_FILE"
print_success "Updated .env file"

# Step 4: Test authentication
print_info "Testing authentication..."

# Test 1: Can we authenticate?
if gcloud auth activate-service-account --key-file="$KEY_FILE" &>/dev/null; then
    print_success "Successfully authenticated with service account"
else
    print_error "Failed to authenticate with service account"
    exit 1
fi

# Test 2: Can we access the project?
if gcloud projects describe "$PROJECT_ID" &>/dev/null; then
    print_success "Successfully accessed project: $PROJECT_ID"
else
    print_error "Cannot access project: $PROJECT_ID"
    exit 1
fi

# Test 3: Can we access BigQuery?
if bq ls --project_id="$PROJECT_ID" &>/dev/null; then
    print_success "Successfully accessed BigQuery"

    # Check if dataset exists
    if bq show "${PROJECT_ID}:trading_signals" &>/dev/null; then
        print_success "Trading signals dataset exists and is accessible"
    else
        print_warning "Trading signals dataset not found (Terraform will create it)"
    fi
else
    print_error "Cannot access BigQuery"
    exit 1
fi

# Test 4: Check IAM permissions
print_info "Checking service account permissions..."
ROLES=$(gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --format="value(bindings.role)")

if [ -n "$ROLES" ]; then
    print_success "Service account has the following roles:"
    echo "$ROLES" | sed 's/^/  - /'
else
    print_warning "Service account has no project-level roles"
fi

# Summary
echo
echo -e "${GREEN}✅ Authentication setup complete!${NC}"
echo
echo "You can now run Terraform commands:"
echo "  cd terraform"
echo "  terraform init -backend-config=backend.conf"
echo "  terraform plan -var-file=terraform.tfvars"
echo "  terraform apply -var-file=terraform.tfvars"
echo
echo "The following environment variable is set for this session:"
echo "  GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE"
echo
echo "To use in a new terminal, run:"
echo "  source .env"
echo
print_info "Service account: $SERVICE_ACCOUNT_EMAIL"
print_info "Project: $PROJECT_ID"

# Optional: Show command to run Terraform
echo
read -p "Do you want to run Terraform now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd terraform
    print_info "Running terraform plan..."
    terraform plan -var-file=terraform.tfvars
fi