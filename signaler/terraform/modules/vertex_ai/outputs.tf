# Outputs for Vertex AI module

output "metadata_store_name" {
  value       = google_vertex_ai_metadata_store.trading_signals.name
  description = "Name of the metadata store"
}

output "metadata_store_id" {
  value       = google_vertex_ai_metadata_store.trading_signals.id
  description = "Full resource ID of the metadata store"
}

output "tensorboard_name" {
  value       = google_vertex_ai_tensorboard.experiments.name
  description = "Name of the Tensorboard instance"
}

output "tensorboard_id" {
  value       = google_vertex_ai_tensorboard.experiments.id
  description = "Full resource ID of the Tensorboard instance"
}

output "dataset_id" {
  value       = google_vertex_ai_dataset.trading_signals.id
  description = "ID of the Vertex AI dataset"
}

output "endpoint_id" {
  value       = google_vertex_ai_endpoint.prediction.id
  description = "ID of the prediction endpoint"
}