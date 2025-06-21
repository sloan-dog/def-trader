# Cloud Run Module

This module manages Cloud Run services for the Trading Signal System.

## Usage

```hcl
module "cloud_run" {
  source = "./modules/cloud_run"

  project_id      = var.project_id
  region          = var.region
  service_account = google_service_account.trading_system.email

  services = {
    daily_ingestion = {
      name   = "daily-ingestion"
      image  = "gcr.io/${var.project_id}/daily-ingestion:latest"
      cpu    = "2"
      memory = "4Gi"
      env_vars = {
        GCP_PROJECT_ID = var.project_id
        BQ_DATASET     = var.bigquery_dataset
      }
      timeout       = 3600
      concurrency   = 100
      min_instances = "0"
      max_instances = "10"
    }
  }
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|----------|
| project_id | GCP Project ID | `string` | - | yes |
| region | GCP region for Cloud Run services | `string` | - | yes |
| service_account | Service account email for Cloud Run services | `string` | - | yes |
| services | Map of Cloud Run services to deploy | `map(object)` | - | yes |

## Outputs

| Name | Description |
|------|-------------|
| service_urls | URLs of the deployed Cloud Run services |
| service_names | Names of the deployed Cloud Run services |

## Service Configuration

Each service in the `services` map can have the following attributes:

- `name` (required): Name of the Cloud Run service
- `image` (required): Docker image to deploy
- `cpu` (required): CPU allocation (e.g., "1", "2", "4")
- `memory` (required): Memory allocation (e.g., "512Mi", "2Gi", "4Gi")
- `env_vars` (required): Map of environment variables
- `timeout` (optional): Request timeout in seconds (default: 3600)
- `concurrency` (optional): Max concurrent requests per instance (default: 100)
- `min_instances` (optional): Minimum number of instances (default: "0")
- `max_instances` (optional): Maximum number of instances (default: "10")