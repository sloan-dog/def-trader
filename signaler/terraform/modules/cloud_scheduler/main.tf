# Cloud Scheduler module for Trading Signal System

# Cloud Scheduler jobs
resource "google_cloud_scheduler_job" "jobs" {
  for_each = var.jobs

  name        = each.value.name
  description = lookup(each.value, "description", "Scheduled job for ${each.value.name}")
  schedule    = each.value.schedule
  time_zone   = each.value.timezone
  region      = var.region

  # Always include retry_config with defaults
  retry_config {
    retry_count          = coalesce(lookup(each.value, "retry_count", null), 3)
    max_retry_duration   = coalesce(lookup(each.value, "max_retry_duration", null), "600s")
    min_backoff_duration = coalesce(lookup(each.value, "min_backoff_duration", null), "5s")
    max_backoff_duration = coalesce(lookup(each.value, "max_backoff_duration", null), "600s")
    max_doublings        = 5
  }

  http_target {
    uri         = each.value.target_url
    http_method = lookup(each.value, "http_method", "POST")

    headers = merge(
      {
        "Content-Type" = "application/json"
      },
      lookup(each.value, "headers", {})
    )

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