# Cloud Run module for Trading Signal System

# Cloud Run services
resource "google_cloud_run_service" "services" {
  for_each = var.services

  name     = each.value.name
  location = var.region

  template {
    spec {
      service_account_name = var.service_account

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
          value_from {
            secret_key_ref {
              name = "alpha-vantage-api-key"
              key  = "latest"
            }
          }
        }
      }

      # Timeout
      timeout_seconds = lookup(each.value, "timeout", 3600)

      # Concurrency
      container_concurrency = lookup(each.value, "concurrency", 100)
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = lookup(each.value, "min_instances", "0")
        "autoscaling.knative.dev/maxScale" = lookup(each.value, "max_instances", "10")
        "run.googleapis.com/cpu-throttling" = "false"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  autogenerate_revision_name = true

  lifecycle {
    ignore_changes = [
      template[0].metadata[0].annotations["run.googleapis.com/client-name"],
      template[0].metadata[0].annotations["run.googleapis.com/client-version"],
    ]
  }
}

# IAM policy for services
resource "google_cloud_run_service_iam_member" "invoker" {
  for_each = var.services

  service  = google_cloud_run_service.services[each.key].name
  location = google_cloud_run_service.services[each.key].location
  role     = "roles/run.invoker"

  # Allow public access for API service, restrict others
  member = each.key == "prediction_service" ? "allUsers" : "serviceAccount:${var.service_account}"
}

# Outputs
output "service_urls" {
  value = {
    for k, v in google_cloud_run_service.services : k => v.status[0].url
  }
  description = "URLs of the deployed Cloud Run services"
}

output "service_names" {
  value = {
    for k, v in google_cloud_run_service.services : k => v.name
  }
  description = "Names of the deployed Cloud Run services"
}