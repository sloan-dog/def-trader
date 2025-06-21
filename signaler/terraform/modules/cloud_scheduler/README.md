# Cloud Scheduler Module

This module manages Cloud Scheduler jobs for the Trading Signal System.

## Usage

```hcl
module "cloud_scheduler" {
  source = "./modules/cloud_scheduler"

  project_id      = var.project_id
  region          = var.region
  service_account = google_service_account.trading_system.email

  jobs = {
    daily_ingestion = {
      name       = "daily-data-ingestion"
      schedule   = "0 18 * * MON-FRI"  # 6 PM EST on weekdays
      timezone   = "America/New_York"
      target_url = module.cloud_run.service_urls["daily_ingestion"]
    }
  }
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|----------|
| project_id | GCP Project ID | `string` | - | yes |
| region | GCP region for Cloud Scheduler jobs | `string` | - | yes |
| service_account | Service account email for Cloud Scheduler jobs | `string` | - | yes |
| jobs | Map of Cloud Scheduler jobs to create | `map(object)` | - | yes |

## Outputs

| Name | Description |
|------|-------------|
| job_names | Names of the created scheduler jobs |
| job_schedules | Schedules of the created jobs |

## Job Configuration

Each job in the `jobs` map can have the following attributes:

- `name` (required): Name of the scheduler job
- `schedule` (required): Cron schedule expression
- `timezone` (required): Timezone for the schedule
- `target_url` (required): HTTP endpoint to invoke
- `description` (optional): Job description
- `http_method` (optional): HTTP method (default: "POST")
- `headers` (optional): HTTP headers map
- `body` (optional): Request body as string
- `retry_count` (optional): Number of retries (default: 3)
- `max_retry_duration` (optional): Max retry duration (default: "600s")
- `min_backoff_duration` (optional): Min backoff duration (default: "5s")
- `max_backoff_duration` (optional): Max backoff duration (default: "600s")

## Schedule Examples

- `"0 18 * * MON-FRI"` - 6 PM every weekday
- `"0 2 * * SUN"` - 2 AM every Sunday
- `"*/15 * * * *"` - Every 15 minutes
- `"0 */4 * * *"` - Every 4 hours