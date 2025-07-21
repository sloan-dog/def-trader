# Cloud Run Jobs module for Trading Signal System
# This version uses placeholder images initially but ignores changes after creation

# Cloud Run Jobs
resource "google_cloud_run_v2_job" "jobs" {
  for_each = var.jobs

  name     = each.value.name
  location = var.region

  template {
    template {
      service_account = var.service_account

      containers {
        image = each.value.image

        resources {
          limits = {
            cpu    = each.value.cpu
            memory = each.value.memory
          }
        }

        # Environment variables
        dynamic "env" {
          for_each = each.value.env_vars
          content {
            name  = env.key
            value = env.value
          }
        }

        # Add secret environment variables if needed
        env {
          name = "ALPHA_VANTAGE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = "alpha-vantage-api-key"
              version = "latest"
            }
          }
        }
      }

      # Timeout
      timeout = lookup(each.value, "timeout", "3600s")
    }
  }

  lifecycle {
    ignore_changes = [
      # Ignore image changes after initial creation
      template[0].template[0].containers[0].image,
    ]
  }
}

# IAM policy for jobs
resource "google_cloud_run_v2_job_iam_member" "invoker" {
  for_each = var.jobs

  name     = google_cloud_run_v2_job.jobs[each.key].name
  location = google_cloud_run_v2_job.jobs[each.key].location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${var.service_account}"
}

# Outputs
output "job_names" {
  value = {
    for k, v in google_cloud_run_v2_job.jobs : k => v.name
  }
  description = "Names of the deployed Cloud Run Jobs"
}

output "job_ids" {
  value = {
    for k, v in google_cloud_run_v2_job.jobs : k => v.id
  }
  description = "IDs of the deployed Cloud Run Jobs"
} 