#!/bin/bash

# Check what Terraform resources exist and what would be created with full apply

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

cd "$(dirname "$0")/../terraform"

echo -e "${BLUE}🔍 Terraform Resource Status Check${NC}"
echo "================================="
echo

# Check current resources
echo -e "${BLUE}Current Resources:${NC}"
CURRENT_RESOURCES=$(terraform state list 2>/dev/null | wc -l)
if [ "$CURRENT_RESOURCES" -eq 0 ]; then
    echo "  No resources currently managed by Terraform"
else
    echo "  Total resources: $CURRENT_RESOURCES"
    echo
    echo "  BigQuery:"
    terraform state list | grep -E "(bigquery|dataset)" | sed 's/^/    - /' || echo "    (none)"
    echo
    echo "  Service Account:"
    terraform state list | grep -E "service_account" | sed 's/^/    - /' || echo "    (none)"
    echo
    echo "  Secrets:"
    terraform state list | grep -E "secret" | sed 's/^/    - /' || echo "    (none)"
    echo
    echo "  Storage:"
    terraform state list | grep -E "storage" | sed 's/^/    - /' || echo "    (none)"
fi

echo
echo -e "${BLUE}Resources that would be added with full apply:${NC}"

# Run a plan to see what would be created
terraform plan -var-file=terraform.tfvars -detailed-exitcode -out=/tmp/tfplan-check > /tmp/tfplan-output.txt 2>&1
PLAN_EXIT_CODE=$?

if [ $PLAN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}  ✓ Infrastructure is up to date - no changes needed${NC}"
elif [ $PLAN_EXIT_CODE -eq 2 ]; then
    # Changes are needed
    echo "  The following resources would be created:"
    echo

    # Extract resources to be created
    grep "will be created" /tmp/tfplan-output.txt | sed 's/.*# /  - /' | sed 's/ will be created//' || true

    # Count resources
    TO_CREATE=$(grep -c "will be created" /tmp/tfplan-output.txt || echo "0")
    echo
    echo -e "${YELLOW}  Total resources to create: $TO_CREATE${NC}"

    # Categorize what would be created
    echo
    echo "  By category:"
    grep "will be created" /tmp/tfplan-output.txt > /tmp/resources-to-create.txt || true

    if grep -q "cloud_run" /tmp/resources-to-create.txt; then
        echo -e "    ${YELLOW}•${NC} Cloud Run services"
    fi
    if grep -q "scheduler" /tmp/resources-to-create.txt; then
        echo -e "    ${YELLOW}•${NC} Cloud Scheduler jobs"
    fi
    if grep -q "monitoring" /tmp/resources-to-create.txt; then
        echo -e "    ${YELLOW}•${NC} Monitoring and alerting"
    fi
    if grep -q "vertex_ai" /tmp/resources-to-create.txt; then
        echo -e "    ${YELLOW}•${NC} Vertex AI resources"
    fi
else
    echo -e "${RED}  Error running terraform plan${NC}"
fi

# Cleanup
rm -f /tmp/tfplan-check /tmp/tfplan-output.txt /tmp/resources-to-create.txt

echo
echo -e "${BLUE}Summary:${NC}"
if [ "$CURRENT_RESOURCES" -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} You have the minimal resources for local development"
    echo -e "  ${BLUE}ℹ${NC} Run 'terraform apply -var-file=terraform.tfvars' when ready for full deployment"
else
    echo -e "  ${YELLOW}!${NC} No resources created yet"
    echo -e "  ${BLUE}ℹ${NC} Run './terraform_minimal.sh' to create essential resources"
fi