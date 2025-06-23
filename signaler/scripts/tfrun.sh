#!/bin/bash

# Terraform wrapper script that ensures proper authentication
# Usage: ./tfrun.sh [any terraform/tofu command]
# Example: ./tfrun.sh plan
#          ./tfrun.sh apply -target=module.bigquery

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
KEY_FILE="$HOME/trading-system-key.json"

# Check if key exists
if [ ! -f "$KEY_FILE" ]; then
    echo -e "❌ Service account key not found at: $KEY_FILE"
    echo "   Run ./setup_terraform_auth.sh first"
    exit 1
fi

# Set authentication
export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"

# Load .env if exists
if [ -f ".env" ]; then
    source .env
fi

# Determine if using terraform or tofu
TF_CMD="tofu"
echo -e "${BLUE}Using OpenTofu${NC}"

# Change to terraform directory
cd ../terraform

# If no arguments, show help
if [ $# -eq 0 ]; then
    echo "Usage: $0 <terraform command>"
    echo
    echo "Common commands:"
    echo "  $0 init                    # Initialize terraform"
    echo "  $0 plan                    # Show what would change"
    echo "  $0 apply                   # Apply changes"
    echo "  $0 apply -target=module.bigquery  # Apply only BigQuery module"
    echo "  $0 destroy                 # Destroy all resources"
    echo "  $0 state list              # List resources in state"
    echo
    echo "Using service account: $(jq -r '.client_email' "$KEY_FILE" 2>/dev/null || echo 'unknown')"
    exit 0
fi

# Check if this is an apply command with a plan file
# A plan file is being used if:
# 1. The command is "apply"
# 2. There's a second argument that is a file (not a flag starting with -)
# 3. The file exists
if [[ "$1" == "apply" ]] && [[ $# -ge 2 ]] && [[ "$2" != -* ]] && [[ -f "$2" ]]; then
    # Applying a plan file - don't add -var-file
    echo -e "${GREEN}Applying saved plan: $2${NC}"
    echo
    $TF_CMD "$@"
else
    # Normal command - add -var-file
    echo -e "${GREEN}Running: $TF_CMD $@ -var-file=terraform.tfvars${NC}"
    echo
    $TF_CMD "$@" -var-file=terraform.tfvars
fi