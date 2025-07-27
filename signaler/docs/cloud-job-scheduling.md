# Cloud Job Scheduling Guide

## Overview

This guide explains how to set up and schedule Cloud Run Jobs with arguments, particularly for the backfill job that needs to run on different schedules with different parameters.

## Current Setup

### Cloud Run Jobs
- **daily-ingestion-job**: Fetches latest market data
- **backfill-job**: Backfills historical data with configurable parameters

### Scheduling Methods

#### 1. Cloud Scheduler (via Terraform)

Currently configured schedules:
- **Daily Ingestion**: `0 18 * * MON-FRI` (6 PM EST weekdays)
- **Hourly Backfill**: `0 * * * *` (Every hour)

#### 2. Manual Execution with Arguments

Execute jobs manually with specific arguments:

```bash
# Basic backfill from a specific date
gcloud run jobs execute backfill-job \
  --region=us-central1 \
  --args="--start-date=1995-01-01"

# Backfill with specific data types
gcloud run jobs execute backfill-job \
  --region=us-central1 \
  --args="--start-date=2020-01-01,--end-date=2024-01-01,--data-types=ohlcv,indicators"

# Full historical backfill
gcloud run jobs execute backfill-job \
  --region=us-central1 \
  --args="--start-date=1995-01-01,--data-types=all,--batch-size=20"
```

## Setting Up Scheduled Jobs with Arguments

### Option 1: Multiple Cloud Scheduler Jobs

Create different scheduler jobs for different backfill strategies:

```hcl
# In terraform/main.tf, add to cloud_scheduler module:

module "cloud_scheduler" {
  # ... existing config ...
  
  jobs = {
    # ... existing jobs ...
    
    # Weekly full backfill
    weekly_full_backfill = {
      name        = "weekly-full-backfill"
      schedule    = "0 3 * * SUN"  # 3 AM EST on Sundays
      timezone    = "America/New_York"
      target_url  = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/backfill-job/executions"
      description = "Weekly full data backfill"
      http_method = "POST"
      body = jsonencode({
        overrides = {
          containerOverrides = [{
            args = ["--start-date=2020-01-01", "--data-types=all"]
          }]
        }
      })
    }
    
    # Monthly historical backfill
    monthly_historical_backfill = {
      name        = "monthly-historical-backfill"
      schedule    = "0 4 1 * *"  # 4 AM EST on 1st of month
      timezone    = "America/New_York"
      target_url  = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/backfill-job/executions"
      description = "Monthly historical data backfill"
      http_method = "POST"
      body = jsonencode({
        overrides = {
          containerOverrides = [{
            args = ["--start-date=1995-01-01", "--end-date=2000-01-01", "--data-types=ohlcv,macro"]
          }]
        }
      })
    }
  }
}
```

### Option 2: Using gcloud CLI to Create Schedulers

Create scheduler jobs directly with gcloud:

```bash
# Create weekly backfill scheduler
gcloud scheduler jobs create http weekly-backfill \
  --location=us-central1 \
  --schedule="0 3 * * SUN" \
  --time-zone="America/New_York" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/trading-signals-420-69/jobs/backfill-job/executions" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{
    "overrides": {
      "containerOverrides": [{
        "args": ["--start-date=2020-01-01", "--data-types=all"]
      }]
    }
  }' \
  --oidc-service-account-email=trading-system@trading-signals-420-69.iam.gserviceaccount.com

# Create monthly historical backfill scheduler
gcloud scheduler jobs create http monthly-historical-backfill \
  --location=us-central1 \
  --schedule="0 4 1 * *" \
  --time-zone="America/New_York" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/trading-signals-420-69/jobs/backfill-job/executions" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{
    "overrides": {
      "containerOverrides": [{
        "args": ["--start-date=1995-01-01", "--end-date=2000-01-01", "--data-types=ohlcv,macro", "--batch-size=50"]
      }]
    }
  }' \
  --oidc-service-account-email=trading-system@trading-signals-420-69.iam.gserviceaccount.com
```

## Backfill Strategy Recommendations

### Initial Historical Load (1995-present)
```bash
# Run once to populate all historical data
gcloud run jobs execute backfill-job \
  --region=us-central1 \
  --args="--start-date=1995-01-01,--data-types=all,--batch-size=50"
```

### Regular Schedules

1. **Hourly** (already configured)
   - Recent data updates
   - No args needed (uses defaults for last 24 hours)

2. **Daily** (6 PM EST weekdays)
   - End-of-day market data
   - Args: `--data-types=ohlcv,indicators`

3. **Weekly** (Sunday 3 AM)
   - Full week reconciliation
   - Args: `--start-date={1 week ago},--data-types=all`

4. **Monthly** (1st of month, 4 AM)
   - Macro data updates
   - Args: `--data-types=macro,temporal`

## Monitoring Scheduled Jobs

### View scheduled jobs
```bash
gcloud scheduler jobs list --location=us-central1
```

### View job executions
```bash
gcloud run jobs executions list --job=backfill-job --region=us-central1
```

### View logs
```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=backfill-job" \
  --project=trading-signals-420-69 \
  --limit=50
```

## Troubleshooting

### Common Issues

1. **Authentication errors**
   - Ensure service account has `roles/run.invoker` permission
   - Check OIDC token configuration

2. **Argument parsing errors**
   - Use comma-separated args format: `--arg1=value1,--arg2=value2`
   - Check Dockerfile CMD configuration

3. **Timeout errors**
   - Increase timeout in Cloud Run Job configuration
   - Consider breaking large backfills into smaller chunks

### Testing Arguments Locally

```bash
# Test the backfill job locally
python -m src.jobs.backfill_job \
  --start-date=2023-01-01 \
  --end-date=2023-12-31 \
  --data-types=ohlcv,indicators \
  --batch-size=10
```