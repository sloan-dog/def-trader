# Backfill Job Deployment Guide for Google Cloud Run

## Building and Pushing the Docker Image

```bash
# Build the Docker image
docker build -f docker/Dockerfile.backfill.job -t gcr.io/YOUR_PROJECT_ID/backfill-job:latest .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/backfill-job:latest
```

## Creating the Cloud Run Job

```bash
# Create the job without any default arguments
gcloud run jobs create backfill-job \
  --image=gcr.io/YOUR_PROJECT_ID/backfill-job:latest \
  --region=YOUR_REGION \
  --max-retries=1 \
  --parallelism=1 \
  --task-timeout=3600
```

## Executing the Job with Different Backfill Types

### Hourly Backfill
```bash
gcloud run jobs execute backfill-job \
  --region=YOUR_REGION \
  --args="--type,hourly"
```

### Daily Backfill
```bash
gcloud run jobs execute backfill-job \
  --region=YOUR_REGION \
  --args="--type,daily"
```

### Weekly Backfill
```bash
gcloud run jobs execute backfill-job \
  --region=YOUR_REGION \
  --args="--type,weekly"
```

### Historical Backfill
```bash
gcloud run jobs execute backfill-job \
  --region=YOUR_REGION \
  --args="--type,historical,--start-year,2020,--end-year,2023,--data-types,ohlcv,--data-types,macro,--batch-size,20"
```

## Important Notes

1. **Arguments Format**: Arguments must be comma-separated in the `--args` parameter
2. **Multiple Values**: For parameters that accept multiple values (like `--data-types`), repeat the parameter
3. **No Quotes**: Don't wrap individual arguments in quotes
4. **Shell Script**: The entrypoint.sh script handles proper argument passing to the Python module

## Scheduling Jobs

To schedule regular backfills, use Cloud Scheduler:

```bash
# Schedule hourly backfill
gcloud scheduler jobs create http hourly-backfill \
  --location=YOUR_REGION \
  --schedule="0 * * * *" \
  --uri="https://YOUR_REGION-run.googleapis.com/v2/projects/YOUR_PROJECT_ID/locations/YOUR_REGION/jobs/backfill-job:run" \
  --http-method=POST \
  --oauth-service-account-email=YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --message-body='{"overrides":{"containerOverrides":[{"args":["--type","hourly"]}]}}'

# Schedule daily backfill
gcloud scheduler jobs create http daily-backfill \
  --location=YOUR_REGION \
  --schedule="0 2 * * *" \
  --uri="https://YOUR_REGION-run.googleapis.com/v2/projects/YOUR_PROJECT_ID/locations/YOUR_REGION/jobs/backfill-job:run" \
  --http-method=POST \
  --oauth-service-account-email=YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --message-body='{"overrides":{"containerOverrides":[{"args":["--type","daily"]}]}}'
```

## Troubleshooting

If you still see the error "terminated: Application failed to start: "--type=historical" not found", check:

1. Ensure the entrypoint.sh script is executable in the Docker image
2. Verify the script is copied to the correct location (/app/entrypoint.sh)
3. Check Cloud Run logs for more detailed error messages:
   ```bash
   gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=backfill-job" --limit=50
   ```