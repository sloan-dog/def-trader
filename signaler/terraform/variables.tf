# Variables for Trading Signal System infrastructure

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for resources"
  type        = string
  default     = "us-central1-a"
}

variable "bigquery_dataset" {
  description = "BigQuery dataset name"
  type        = string
  default     = "trading_signals"
}

variable "bigquery_location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API key"
  type        = string
  sensitive   = true
}

variable "alert_email" {
  description = "Email address for alerts"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "enable_apis" {
  description = "Whether to enable required Google APIs"
  type        = bool
  default     = true
}

variable "cloud_run_config" {
  description = "Configuration for Cloud Run services"
  type = object({
    cpu                 = string
    memory              = string
    max_instances       = number
    min_instances       = number
    timeout_seconds     = number
    concurrency         = number
  })
  default = {
    cpu                 = "2"
    memory              = "4Gi"
    max_instances       = 10
    min_instances       = 0
    timeout_seconds     = 3600
    concurrency         = 100
  }
}

variable "vertex_ai_config" {
  description = "Configuration for Vertex AI training"
  type = object({
    machine_type        = string
    accelerator_type    = string
    accelerator_count   = number
    boot_disk_size_gb   = number
    training_image_uri  = string
  })
  default = {
    machine_type        = "n1-standard-8"
    accelerator_type    = "NVIDIA_TESLA_T4"
    accelerator_count   = 1
    boot_disk_size_gb   = 100
    training_image_uri  = null
  }
}

variable "scheduler_config" {
  description = "Configuration for Cloud Scheduler jobs"
  type = object({
    time_zone           = string
    retry_count         = number
    max_retry_duration  = string
    min_backoff_duration = string
    max_backoff_duration = string
  })
  default = {
    time_zone           = "America/New_York"
    retry_count         = 3
    max_retry_duration  = "600s"
    min_backoff_duration = "5s"
    max_backoff_duration = "600s"
  }
}

variable "data_retention_days" {
  description = "Number of days to retain data in BigQuery"
  type        = number
  default     = 365
}

variable "enable_monitoring" {
  description = "Whether to enable monitoring and alerting"
  type        = bool
  default     = true
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server URI"
  type        = string
  default     = ""
}

variable "docker_registry" {
  description = "Docker registry for container images"
  type        = string
  default     = "gcr.io"
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    project     = "trading-signals"
    managed_by  = "terraform"
    environment = "production"
  }
}