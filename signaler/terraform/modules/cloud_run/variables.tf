# Variables for Cloud Run module

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run services"
  type        = string
}

variable "service_account" {
  description = "Service account email for Cloud Run services"
  type        = string
}

variable "services" {
  description = "Map of Cloud Run services to deploy"
  type = map(object({
    name          = string
    image         = string
    cpu           = string
    memory        = string
    env_vars      = map(string)
    timeout       = optional(number)
    concurrency   = optional(number)
    min_instances = optional(string)
    max_instances = optional(string)
  }))
}