variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "service_account" {
  description = "Service account email for the jobs"
  type        = string
}

variable "jobs" {
  description = "Map of Cloud Run Jobs to create"
  type = map(object({
    name   = string
    image  = string
    cpu    = string
    memory = string
    env_vars = map(string)
    timeout = optional(string, "3600s")
    max_retry_count = optional(number, 3)
    task_count = optional(number, 1)
  }))
} 