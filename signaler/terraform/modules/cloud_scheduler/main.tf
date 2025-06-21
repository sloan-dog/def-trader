# Cloud Scheduler module for Trading Signal System

# Cloud Scheduler jobs
resource "google_cloud_scheduler_job" "jobs" {
  for_each = var.jobs

  name        = each.value.name
  description = lookup(each.value, "description", "Scheduled job for ${each.value.name}")
  schedule    = each.value.schedule
  time_zone   = each.value.timezone
  region      = var.region

  retry_config {
    retry_count          = lookup(each.value, "retry_count", 3)
    max_retry_duration   = lookup(each.value, "max_retry_duration", "600s")
    min_backoff_duration = lookup(each.value, "min_backoff_duration", "5s")
    max_backoff_duration = lookup(each.value, "max_backoff_duration", "600s")
  }

  http_target {
    uri         = each.value.target_url
    http_method = lookup(each.value, "http_method", "POST")

    headers = lookup(each.value, "headers", {
      "Content-Type" = "application/json"
    })

    # Body for the request if needed
    body = lookup(each.value, "body", null) != null ? base64encode(each.value.body) : null

    # OIDC token for authentication
    oidc_token {
      service_account_email = var.service_account
      audience              = each.value.target_url
    }
  }
}

# Outputs
output "job_names" {
  value = {
    for k, v in google_cloud_scheduler_job.jobs : k => v.name
  }
  description = "Names of the created scheduler jobs"
}

output "job_schedules" {
  value = {
    for k, v in google_cloud_scheduler_job.jobs : k => v.schedule
  }
  description = "Schedules of the created jobs"
}