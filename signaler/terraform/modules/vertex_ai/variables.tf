# Variables for Vertex AI module

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Vertex AI resources"
  type        = string
}

variable "service_account" {
  description = "Service account email for Vertex AI"
  type        = string
}

variable "training_config" {
  description = "Configuration for Vertex AI training jobs"
  type = object({
    machine_type        = string
    accelerator_type    = string
    accelerator_count   = number
    boot_disk_size_gb   = number
    training_image_uri  = optional(string)
  })
  default = {
    machine_type        = "n1-standard-8"
    accelerator_type    = "NVIDIA_TESLA_T4"
    accelerator_count   = 1
    boot_disk_size_gb   = 100
    training_image_uri  = null
  }
}

variable "create_workbench_instance" {
  description = "Whether to create a Vertex AI Workbench instance for development"
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels to apply to Vertex AI resources"
  type        = map(string)
  default     = {}
}