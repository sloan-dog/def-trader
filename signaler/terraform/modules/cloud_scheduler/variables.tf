# Variables for Cloud Scheduler module

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Scheduler jobs"
  type        = string
}

variable "service_account" {
  description = "Service account email for Cloud Scheduler jobs"
  type        = string
}

variable "jobs" {
  description = "Map of Cloud Scheduler jobs to create"
  type = map(object({
    name               = string
    schedule           = string
    timezone           = string
    target_url         = string
    description        = optional(string)
    http_method        = optional(string)
    headers            = optional(map(string))
    body              = optional(string)
    retry_count        = optional(number)
    max_retry_duration = optional(string)
    min_backoff_duration = optional(string)
    max_backoff_duration = optional(string)
  }))
}