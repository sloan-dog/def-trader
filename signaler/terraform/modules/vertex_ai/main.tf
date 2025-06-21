# Vertex AI module for Trading Signal System

# Storage bucket for Vertex AI artifacts
resource "google_storage_bucket" "vertex_ai_staging" {
  name     = "${var.project_id}-vertex-ai-staging"
  location = var.region

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Vertex AI Tensorboard
resource "google_vertex_ai_tensorboard" "training_tensorboard" {
  display_name = "Trading Signal Model Training"
  description  = "Tensorboard for tracking model training experiments"
  region       = var.region
}

# Vertex AI Metadata Store (for experiment tracking)
resource "google_vertex_ai_custom_job" "default" {
  name        = "${var.project_id}-metadata-store"
  description = "Metadata store for ML experiments and artifacts"
  region      = var.region
}

# Custom training job configuration (as a local for reuse)
locals {
  training_job_spec = {
    worker_pool_specs = [
      {
        machine_spec = {
          machine_type       = var.training_config.machine_type
          accelerator_type   = var.training_config.accelerator_type
          accelerator_count  = var.training_config.accelerator_count
        }
        replica_count = 1
        disk_spec = {
          boot_disk_type    = "pd-ssd"
          boot_disk_size_gb = var.training_config.boot_disk_size_gb
        }
        container_spec = {
          image_uri = var.training_config.training_image_uri != null ? var.training_config.training_image_uri : "gcr.io/${var.project_id}/trading-system/trainer:latest"
          command = ["python", "-m", "src.jobs.training_job"]
          args = [
            "--experiment-name", "vertex-ai-training",
            "--mlflow-uri", "gs://${google_storage_bucket.vertex_ai_staging.name}/mlflow"
          ]
          env = [
            {
              name  = "GCP_PROJECT_ID"
              value = var.project_id
            },
            {
              name  = "BQ_DATASET"
              value = "trading_signals"
            },
            {
              name  = "MODEL_OUTPUT_PATH"
              value = "gs://${google_storage_bucket.vertex_ai_staging.name}/models"
            }
          ]
        }
      }
    ]

    service_account = var.service_account

    tensorboard = google_vertex_ai_tensorboard.training_tensorboard.name

    base_output_directory = {
      output_uri_prefix = "gs://${google_storage_bucket.vertex_ai_staging.name}/jobs"
    }
  }
}

# IAM permissions for Vertex AI
resource "google_storage_bucket_iam_member" "vertex_ai_staging_access" {
  bucket = google_storage_bucket.vertex_ai_staging.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${var.service_account}"
}

# Workbench Instance (optional - for development)
resource "google_notebooks_instance" "development" {
  count = var.create_workbench_instance ? 1 : 0

  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf-latest-gpu"
  }

  name         = "trading-signal-dev"
  location     = "${var.region}-a"
  machine_type = "n1-standard-4"

  install_gpu_driver = var.training_config.accelerator_count > 0

  dynamic "accelerator_config" {
    for_each = var.training_config.accelerator_count > 0 ? [1] : []
    content {
      type       = var.training_config.accelerator_type
      core_count = var.training_config.accelerator_count
    }
  }

  boot_disk_type    = "PD_SSD"
  boot_disk_size_gb = 100

  no_public_ip    = false
  no_proxy_access = false

  instance_owners = [var.service_account]

  metadata = {
    tensorflow-version = "2.11.0"
    framework          = "TensorFlow:2.11"
  }

  service_account = var.service_account
}

terraform {
  required_providers {
    google = {
      source                = "hashicorp/google"
      version               = "~> 4.0"
      configuration_aliases = [google]
    }
    google-beta = {
      source                = "hashicorp/google-beta"
      version               = "~> 4.0"
      configuration_aliases = [google-beta]
    }
  }
}

# Metadata Store (using beta provider)
resource "google_vertex_ai_metadata_store" "trading_signals" {
  provider = google-beta

  name        = "trading-signals-metadata"
  description = "Metadata store for trading signal ML experiments"
  region      = var.region
}

# Tensorboard instance for experiment visualization
resource "google_vertex_ai_tensorboard" "experiments" {
  provider = google-beta

  display_name = "trading-signals-experiments"
  description  = "Tensorboard for tracking GNN training experiments"
  region       = var.region
}

# Outputs
output "metadata_store_name" {
  value = google_vertex_ai_metadata_store.trading_signals.name
}

# Outputs
output "staging_bucket" {
  value       = google_storage_bucket.vertex_ai_staging.name
  description = "Name of the Vertex AI staging bucket"
}

output "tensorboard_id" {
  value       = google_vertex_ai_tensorboard.training_tensorboard.id
  description = "ID of the Vertex AI Tensorboard"
}

output "tensorboard_name" {
  value       = google_vertex_ai_tensorboard.training_tensorboard.name
  description = "Name of the Vertex AI Tensorboard"
}

output "metadata_store_id" {
  value       = google_vertex_ai_metadata_store.default.id
  description = "ID of the Vertex AI Metadata Store"
}

output "training_job_spec" {
  value       = local.training_job_spec
  description = "Training job specification for use in custom training jobs"
  sensitive   = true
}

output "workbench_instance_name" {
  value       = var.create_workbench_instance ? google_notebooks_instance.development[0].name : null
  description = "Name of the Workbench instance (if created)"
}