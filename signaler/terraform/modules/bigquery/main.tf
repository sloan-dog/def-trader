# BigQuery module - Manages only the dataset, not tables

# Create the dataset
resource "google_bigquery_dataset" "trading_signals" {
  dataset_id    = var.dataset_id
  friendly_name = "Trading Signals Dataset"
  description   = "Dataset for trading signal system data"
  location      = var.location

  # Access controls
  access {
    role          = "OWNER"
    user_by_email = var.service_account_email
  }

  access {
    role          = "OWNER"
    special_group = "projectOwners"
  }

  access {
    role          = "READER"
    special_group = "projectReaders"
  }

  access {
    role          = "WRITER"
    special_group = "projectWriters"
  }

  # Optional: Set default table expiration if needed
  default_table_expiration_ms = var.default_table_expiration_ms

  # Labels
  labels = var.labels

  # Don't delete tables when dataset is deleted (safety)
  delete_contents_on_destroy = false
}

# Outputs
output "dataset_id" {
  value       = google_bigquery_dataset.trading_signals.dataset_id
  description = "The ID of the BigQuery dataset"
}

output "dataset_self_link" {
  value       = google_bigquery_dataset.trading_signals.self_link
  description = "The self link of the BigQuery dataset"
}

output "dataset_location" {
  value       = google_bigquery_dataset.trading_signals.location
  description = "The location of the BigQuery dataset"
}