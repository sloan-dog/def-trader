# Variables for BigQuery module

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "service_account_email" {
  description = "Service account email to use for BigQuery operations"
  type        = string
}

variable "dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
}

variable "location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

variable "default_table_expiration_ms" {
  description = "Default table expiration in milliseconds"
  type        = number
  default     = null  # No expiration by default
}

variable "labels" {
  description = "Labels to apply to BigQuery resources"
  type        = map(string)
  default     = {}
}