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
    daily_ingestion = {
      name   = "daily-ingestion"
      image  = "gcr.io/${var.project_id}/trading-system/daily-ingestion:latest"
      cpu    = "2"
      memory = "4Gi"
      env_vars = {
        GCP_PROJECT_ID = var.project_id
        BQ_DATASET     = var.bigquery_dataset
      }
    }

    prediction_service = {
      name   = "prediction-service"
      image  = "gcr.io/${var.project_id}/trading-system/prediction-service:latest"
      cpu    = "4"
      memory = "8Gi"
      env_vars = {
        GCP_PROJECT_ID = var.project_id
        BQ_DATASET     = var.bigquery_dataset
      }
    }
  }
}

# Cloud Scheduler jobs
module "cloud_scheduler" {
  source = "./modules/cloud_scheduler"

  project_id      = var.project_id
  region          = var.region
  service_account = google_service_account.trading_system.email

  jobs = {
    daily_ingestion = {
      name     = "daily-data-ingestion"
      schedule = "0 18 * * MON-FRI"  # 6 PM EST on weekdays
      timezone = "America/New_York"
      target_url = module.cloud_run.service_urls["daily_ingestion"]
    }

    weekly_training = {
      name     = "weekly-model-training"
      schedule = "0 2 * * SUN"  # 2 AM EST on Sundays
      timezone = "America/New_York"
      target_url = "${module.cloud_run.service_urls["prediction_service"]}/train"
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