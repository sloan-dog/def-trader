# Local Development Setup - Incremental Terraform Approach

This guide explains how to set up just the essential GCP resources for local development, with the ability to expand to full production infrastructure later without conflicts.

## Quick Start

```bash
# 1. Make scripts executable
chmod +x ../scripts/terraform_minimal.sh
chmod +x ../scripts/terraform_check.sh

# 2. Create minimal resources for local development
../scripts/terraform_minimal.sh

# 3. Create service account key
gcloud iam service-accounts keys create ~/trading-system-key.json \
  --iam-account=trading-system@trading-signals-420-69.iam.gserviceaccount.com

# 4. Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/trading-system-key.json

# 5. Test the setup
python ../test_setup.py
```

## What Gets Created

The minimal setup creates only what you need for local development:

### ✅ Created by `terraform_minimal.sh`:
- **BigQuery**: Dataset and all tables
- **Service Account**: `trading-system` with necessary permissions
- **Secret Manager**: Storage for Alpha Vantage API key
- **Storage Bucket**: For model artifacts (optional but useful)
- **APIs**: Enables BigQuery, Storage, and Secret Manager APIs

### ❌ NOT Created (saved for later):
- Cloud Run services
- Cloud Scheduler jobs
- Vertex AI resources
- Monitoring/Alerting
- Additional service accounts

## Workflow

### Phase 1: Local Development
```bash
# Create minimal resources
./scripts/terraform_minimal.sh

# Check what exists
./scripts/terraform_check.sh

# Start developing locally
python -m src.jobs.backfill_job --start-date 2024-01-01 --end-date 2024-01-31
```

### Phase 2: Full Deployment (Later)
```bash
# When ready to deploy everything
terraform apply -var-file=terraform.tfvars

# This will:
# - Skip resources that already exist (BigQuery, etc.)
# - Add Cloud Run, Scheduler, and other production resources
# - No conflicts! Terraform recognizes existing resources
```

## Why This Works

1. **Terraform State**: All resources are tracked in the same state file
2. **Idempotency**: Terraform won't recreate resources that already exist
3. **Incremental**: You can add resources gradually as needed
4. **Cost-Effective**: Only pay for what you use during development

## Checking Resource Status

```bash
# See what resources currently exist
terraform state list

# See what would be created with full apply
./scripts/terraform_check.sh

# Get more details about a specific resource
terraform state show module.bigquery
```

## Common Commands

```bash
# Refresh state (sync with actual GCP resources)
terraform refresh -var-file=terraform.tfvars

# Destroy only specific resources (careful!)
terraform destroy -target=google_storage_bucket.models -var-file=terraform.tfvars

# Import existing resources (if created outside Terraform)
terraform import google_bigquery_dataset.trading_signals trading-signals-420-69:trading_signals
```

## Troubleshooting

### "Resource already exists" error
```bash
# Import the existing resource into Terraform state
terraform import <resource_type>.<resource_name> <resource_id>
```

### State lock issues
```bash
# Force unlock (use carefully)
terraform force-unlock <lock_id>
```

### Verify BigQuery access
```bash
bq ls
bq show trading_signals
```

## Cost Optimization

During development, you're only paying for:
- BigQuery storage: ~$0.02/GB/month
- BigQuery queries: $5/TB processed
- Secret Manager: $0.06/secret/month
- Storage bucket: ~$0.02/GB/month

The expensive resources (Cloud Run, Vertex AI) are created later when needed.