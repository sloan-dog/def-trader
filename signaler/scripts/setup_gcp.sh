#!/bin/bash

# Simple GCP Setup for Trading Signal System
set -e

# Check for project name argument
if [[ $# -eq 0 ]]; then
    echo "❌ Error: Project name required"
    echo "Usage: $0 <project-name>"
    echo "Example: $0 trading-signals-420-69"
    exit 1
fi

PROJECT_ID="$1"

# Validate project ID format (GCP requirements)
if ! [[ "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
    echo "❌ Error: Invalid project ID format"
    echo "Project IDs must:"
    echo "  - Be 6-30 characters long"
    echo "  - Start with a lowercase letter"
    echo "  - Contain only lowercase letters, numbers, and hyphens"
    echo "  - End with a letter or number"
    exit 1
fi

echo "🚀 Trading Signal System - GCP Setup"
echo "📋 Project ID: $PROJECT_ID"
echo

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated. Run: gcloud auth login"
    exit 1
fi

echo "✅ Authenticated as: $(gcloud auth list --filter=status:ACTIVE --format="value(account)")"
echo

# Check if project exists
echo "🔍 Checking if project exists..."
if gcloud projects describe "$PROJECT_ID" &>/dev/null; then
    echo "✅ Project '$PROJECT_ID' already exists"
    SKIP_PROJECT_CREATE=true
    SKIP_BILLING_SELECTION=true

    # Get current billing info
    BILLING_INFO=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || echo "")
    if [[ -n "$BILLING_INFO" ]]; then
        BILLING_ID=$(echo "$BILLING_INFO" | sed 's|billingAccounts/||')
        BILLING_NAME="Existing"
        echo "✅ Billing already configured"
    else
        echo "⚠️  Warning: Project exists but has no billing account"
        SKIP_BILLING_SELECTION=false
    fi
else
    echo "📝 Project '$PROJECT_ID' does not exist - will create"
    SKIP_PROJECT_CREATE=false
    SKIP_BILLING_SELECTION=false
fi

# Only show billing account selection if needed
if [[ "$SKIP_BILLING_SELECTION" != "true" ]]; then
    # Show ALL billing accounts first
    echo
    echo "=== Your Billing Accounts ==="
    gcloud billing accounts list
    echo

    # Get active billing accounts and show them clearly
    echo "=== Choose Your Billing Account ==="

    # Alternative to mapfile - read accounts into arrays
    ACCOUNT_IDS=()
    ACCOUNT_NAMES=()
    while IFS=$'\t' read -r ID NAME; do
        ACCOUNT_IDS+=("$ID")
        ACCOUNT_NAMES+=("$NAME")
    done < <(gcloud billing accounts list --filter="open=true" --format="value(name,displayName)")

    # Check if we have any accounts
    if [[ ${#ACCOUNT_IDS[@]} -eq 0 ]]; then
        echo "❌ No active billing accounts found!"
        echo "Activate one at: https://console.cloud.google.com/billing"
        exit 1
    fi

    # Show numbered options clearly
    for i in "${!ACCOUNT_IDS[@]}"; do
        echo "$((i+1)). ${ACCOUNT_NAMES[i]} (${ACCOUNT_IDS[i]})"
    done
    echo

    # Get user selection
    while true; do
        read -p "Select billing account (1-${#ACCOUNT_IDS[@]}): " CHOICE
        if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [[ $CHOICE -ge 1 ]] && [[ $CHOICE -le ${#ACCOUNT_IDS[@]} ]]; then
            BILLING_ID="${ACCOUNT_IDS[$((CHOICE-1))]}"
            BILLING_NAME="${ACCOUNT_NAMES[$((CHOICE-1))]}"
            echo "✅ Selected: $BILLING_NAME ($BILLING_ID)"
            break
        else
            echo "❌ Invalid choice. Enter 1-${#ACCOUNT_IDS[@]}"
        fi
    done
fi

# Configuration summary
echo
echo "📋 Configuration Summary:"
echo "  Project ID: $PROJECT_ID"
if [[ -n "$BILLING_NAME" ]]; then
    echo "  Billing: $BILLING_NAME ($BILLING_ID)"
fi
echo "  Action: $(if [[ "$SKIP_PROJECT_CREATE" == "true" ]]; then echo "Use existing project"; else echo "Create new project"; fi)"
echo

read -p "Continue? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create project if needed
if [[ "$SKIP_PROJECT_CREATE" != "true" ]]; then
    echo
    echo "🏗️  Creating project..."
    gcloud projects create "$PROJECT_ID" --name="Trading Signals Dev"
    gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ID"
else
    echo
    echo "📂 Using existing project..."
    # Link billing if needed
    if [[ "$SKIP_BILLING_SELECTION" != "true" ]] && [[ -n "$BILLING_ID" ]]; then
        echo "🔗 Linking billing account..."
        gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ID"
    fi
fi

# Always set as current project
gcloud config set project "$PROJECT_ID"

# Verify billing
if [[ "$SKIP_PROJECT_CREATE" != "true" ]] || [[ -z "$BILLING_ID" ]]; then
    if [[ $(gcloud billing projects describe "$PROJECT_ID" --format="value(billingEnabled)") == "True" ]]; then
        echo "✅ Billing linked successfully"
    else
        echo "❌ Billing link failed"
        exit 1
    fi
fi

# Enable APIs (idempotent - safe to run multiple times)
echo "🔧 Enabling APIs..."
APIS=(
    bigquery.googleapis.com
    storage.googleapis.com
    secretmanager.googleapis.com
    run.googleapis.com
    cloudbuild.googleapis.com
    containerregistry.googleapis.com
    aiplatform.googleapis.com
    cloudscheduler.googleapis.com
)

# Check which APIs need to be enabled
APIS_TO_ENABLE=()
for api in "${APIS[@]}"; do
    if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" 2>/dev/null | grep -q "$api"; then
        APIS_TO_ENABLE+=("$api")
    fi
done

if [[ ${#APIS_TO_ENABLE[@]} -eq 0 ]]; then
    echo "✅ All required APIs already enabled"
else
    echo "Enabling ${#APIS_TO_ENABLE[@]} APIs..."
    gcloud services enable "${APIS_TO_ENABLE[@]}"
fi

# Create Terraform state bucket if it doesn't exist
echo "📦 Checking Terraform state bucket..."
BUCKET_NAME="gs://${PROJECT_ID}-terraform-state"
if gsutil ls -b "$BUCKET_NAME" &>/dev/null; then
    echo "✅ Terraform state bucket already exists"
else
    echo "Creating Terraform state bucket..."
    gsutil mb -p "$PROJECT_ID" "$BUCKET_NAME"
fi
# Always ensure versioning is on
gsutil versioning set on "$BUCKET_NAME"

# Create backend configuration file
echo "🔧 Creating Terraform backend configuration..."
cat > terraform/backend.conf << EOF
bucket = "${PROJECT_ID}-terraform-state"
prefix = "terraform/state"
EOF

# Create terraform.tfvars
echo "📝 Creating Terraform variables..."
echo
read -p "Enter Alpha Vantage API key (get from https://www.alphavantage.co/support/#api-key): " ALPHA_KEY
if [[ -z "$ALPHA_KEY" ]]; then
    ALPHA_KEY="YOUR_ALPHA_VANTAGE_API_KEY"
fi

USER_EMAIL=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")

cat > terraform/terraform.tfvars << EOF
project_id = "$PROJECT_ID"
region = "us-central1"
bigquery_dataset = "trading_signals"
alpha_vantage_api_key = "$ALPHA_KEY"
alert_email = "$USER_EMAIL"
environment = "dev"
EOF

# Success message
echo
echo "🎉 Setup Complete!"
echo
echo "📋 Summary:"
echo "  Project ID: $PROJECT_ID"
if [[ -n "$BILLING_NAME" ]]; then
    echo "  Billing: $BILLING_NAME"
fi
echo "  Terraform config: ✅"
echo "  Backend config: terraform/backend.conf"
echo "  APIs enabled: ✅"
echo
echo "🚀 Next Steps:"
echo "  1. cd terraform"
echo "  2. terraform init -backend-config=backend.conf"
echo "  3. terraform plan -var-file=terraform.tfvars"
echo "  4. terraform apply -var-file=terraform.tfvars"
echo
echo "💡 After Terraform completes, create service account key:"
echo "  gcloud iam service-accounts keys create ~/trading-system-key.json \\"
echo "    --iam-account=trading-system@${PROJECT_ID}.iam.gserviceaccount.com"
echo
echo "📌 Your project: $PROJECT_ID"
echo
echo "To run this script again for the same project:"
echo "  $0 $PROJECT_ID"