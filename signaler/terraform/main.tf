# Main Terraform configuration for Trading Signal System

terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.0"
    }
  }

  backend "gcs" {
    bucket = "trading-system-terraform-state"
    prefix = "terraform/state"
  }
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Beta provider configuration for Vertex AI features
provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "bigquery.googleapis.com",
    "run.googleapis.com",
    "cloudscheduler.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "compute.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com"
  ])

  service            = each.value
  disable_on_destroy = false
}

# Service Account for the trading system
resource "google_service_account" "trading_system" {
  account_id   = "trading-system"
  display_name = "Trading System Service Account"
  description  = "Service account for trading signal system"
}

# IAM roles for service account
resource "google_project_iam_member" "service_account_roles" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/run.invoker",
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.trading_system.email}"
}

# BigQuery dataset
module "bigquery" {
  source = "./modules/bigquery"

  project_id            = var.project_id
  dataset_id            = var.bigquery_dataset
  location              = var.bigquery_location
  service_account_email = google_service_account.trading_system.email
}

# Vertex AI resources including Metadata Store
module "vertex_ai" {
  source = "./modules/vertex_ai"

  providers = {
    google      = google
    google-beta = google-beta
  }

  project_id      = var.project_id
  region          = var.region
  service_account = google_service_account.trading_system.email

  training_config = {
    machine_type      = "n1-standard-8"
    accelerator_type  = "NVIDIA_TESLA_T4"
    accelerator_count = 1
    boot_disk_size_gb = 100
  }
}

# Cloud Storage buckets
resource "google_storage_bucket" "models" {
  name     = "${var.project_id}-model-registry"
  location = var.region

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "data" {
  name     = "${var.project_id}-trading-data"
  location = var.region

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Parquet data storage bucket
resource "google_storage_bucket" "parquet_data" {
  name     = "${var.project_id}-parquet-data"
  location = var.region
  
  # Enable uniform bucket-level access for better performance
  uniform_bucket_level_access = true
  
  # Storage class for frequently accessed data
  storage_class = "STANDARD"

  # Lifecycle rules for data management
  lifecycle_rule {
    condition {
      age = 365  # Move to NEARLINE after 1 year
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 730  # Move to COLDLINE after 2 years
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  # Versioning for data safety
  versioning {
    enabled = true
  }
  
  # CORS configuration for potential web access
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
  
  # Labels for organization
  labels = {
    environment = var.environment
    purpose     = "time-series-data"
    format      = "parquet"
  }
}

# Secret Manager for API keys
resource "google_secret_manager_secret" "alpha_vantage_key" {
  secret_id = "alpha-vantage-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "alpha_vantage_key" {
  secret      = google_secret_manager_secret.alpha_vantage_key.id
  secret_data = var.alpha_vantage_api_key
}

# Cloud Run services
module "cloud_run" {
  source = "./modules/cloud_run"

  project_id         = var.project_id
  region             = var.region
  service_account    = google_service_account.trading_system.email

  services = {
    prediction_service = {
      name   = "prediction-service"
      # Use Google's hello world image as placeholder
      # CI/CD will update this, and Terraform will ignore changes
      image  = "us-docker.pkg.dev/cloudrun/container/hello"
      cpu    = "4"
      memory = "8Gi"
      env_vars = {
        GCP_PROJECT_ID = var.project_id
        BQ_DATASET     = var.bigquery_dataset
        GCS_BUCKET     = google_storage_bucket.parquet_data.name
      }
    }


  }
}

# Cloud Run Jobs
module "cloud_run_jobs" {
  source = "./modules/cloud_run_jobs"

  project_id      = var.project_id
  region          = var.region
  service_account = google_service_account.trading_system.email

  jobs = {
    daily_ingestion = {
      name   = "daily-ingestion-job"
      # Use Google's hello world image as placeholder
      # CI/CD will update this, and Terraform will ignore changes
      image  = "us-docker.pkg.dev/cloudrun/container/hello"
      cpu    = "4"
      memory = "8Gi"
      env_vars = {
        GCP_PROJECT_ID = var.project_id
        BQ_DATASET     = var.bigquery_dataset
        GCS_BUCKET     = google_storage_bucket.parquet_data.name
      }
      timeout = "7200s"  # 2 hours for ingestion job
    }

    backfill = {
      name   = "backfill-job"
      # Use Google's hello world image as placeholder
      # CI/CD will update this, and Terraform will ignore changes
      image  = "us-docker.pkg.dev/cloudrun/container/hello"
      cpu    = "4"
      memory = "8Gi"
      env_vars = {
        GCP_PROJECT_ID = var.project_id
        BQ_DATASET     = var.bigquery_dataset
        GCS_BUCKET     = google_storage_bucket.parquet_data.name
      }
      timeout = "7200s"  # 2 hours for backfill job
    }

    ohlcv_parquet = {
      name   = "ohlcv-parquet-ingestion-job"
      # Use Google's hello world image as placeholder
      # CI/CD will update this, and Terraform will ignore changes
      image  = "us-docker.pkg.dev/cloudrun/container/hello"
      cpu    = "4"
      memory = "8Gi"
      env_vars = {
        GCP_PROJECT_ID       = var.project_id
        GCS_BUCKET          = google_storage_bucket.parquet_data.name
      }
      timeout = "7200s"  # 2 hours for backfill operations
    }
  }
}

# Cloud Scheduler jobs
module "cloud_scheduler" {
  source = "./modules/cloud_scheduler"

  # Add explicit dependency on Cloud Run modules
  depends_on = [module.cloud_run, module.cloud_run_jobs]

  project_id      = var.project_id
  region          = var.region
  service_account = google_service_account.trading_system.email

  jobs = {
    daily_ingestion = {
      name        = "daily-data-ingestion"
      schedule    = "0 18 * * MON-FRI"  # 6 PM EST on weekdays
      timezone    = "America/New_York"
      # Use Cloud Run Job execution URL
      target_url  = try("https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${module.cloud_run_jobs.job_names["daily_ingestion"]}/executions", "https://placeholder-url")
      description = "Daily market data ingestion job"
      retry_count = 3
      max_retry_duration = "600s"
      http_method = "POST"
      headers = {
        "Content-Type" = "application/json"
      }
    }

    weekly_training = {
      name        = "weekly-model-training"
      schedule    = "0 2 * * SUN"  # 2 AM EST on Sundays
      timezone    = "America/New_York"
      target_url  = try("${module.cloud_run.service_urls["prediction_service"]}/train", "https://placeholder-url")
      description = "Weekly model training job"
      retry_count = 1
      max_retry_duration = "3600s"
    }

    hourly_backfill = {
      name        = "hourly-backfill"
      schedule    = "0 * * * *"  # Every hour
      timezone    = "America/New_York"
      # Use Cloud Run Job execution URL instead of service URL
      target_url  = try("https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${module.cloud_run_jobs.job_names["backfill"]}/executions", "https://placeholder-url")
      description = "Hourly backfill for recent market data"
      retry_count = 3
      max_retry_duration = "1800s"
      http_method = "POST"
      headers = {
        "Content-Type" = "application/json"
      }
    }

    hourly_ohlcv_parquet = {
      name        = "hourly-ohlcv-parquet-update"
      schedule    = "0 * * * *"  # Every hour
      timezone    = "America/New_York"
      # Use Cloud Run Job execution URL for OHLCV Parquet job
      target_url  = try("https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${module.cloud_run_jobs.job_names["ohlcv_parquet"]}/executions", "https://placeholder-url")
      description = "Hourly OHLCV data update to Parquet/GCS"
      retry_count = 3
      max_retry_duration = "1800s"
      http_method = "POST"
      headers = {
        "Content-Type" = "application/json"
      }
      # Pass args for update mode with 2 hour lookback
      body = jsonencode({
        overrides = {
          containerOverrides = [{
            args = ["update", "--lookback", "2"]
          }]
        }
      })
    }
  }
}

# Monitoring and alerting
resource "google_monitoring_alert_policy" "job_failures" {
  display_name = "Trading System Job Failures"
  combiner = "OR"
  conditions {
    display_name = "Job failure rate"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.label.response_code_class!=\"2xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "86400s"
  }
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Notification"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

# Outputs
output "bigquery_dataset" {
  value = module.bigquery.dataset_id
}

output "cloud_run_urls" {
  value = module.cloud_run.service_urls
}

output "service_account_email" {
  value = google_service_account.trading_system.email
}

output "metadata_store_name" {
  value = module.vertex_ai.metadata_store_name
}

output "tensorboard_name" {
  value = module.vertex_ai.tensorboard_name
}

output "parquet_data_bucket" {
  value       = google_storage_bucket.parquet_data.name
  description = "GCS bucket for Parquet time-series data"
}