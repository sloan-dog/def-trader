# Vertex AI module with Metadata Store

terraform {
  required_providers {
    google = {
      source                = "hashicorp/google"
      version               = "~> 4.0"
      configuration_aliases = [google]
    }
    google-beta = {
      source                = "hashicorp/google-beta"
      version               = "~> 4.0"
      configuration_aliases = [google-beta]
    }
  }
}

# Metadata Store for ML experiment tracking
resource "google_vertex_ai_metadata_store" "trading_signals" {
  provider = google-beta

  name        = "trading-signals-metadata"
  description = "Metadata store for trading signal ML experiments"
  region      = var.region
}

# Tensorboard instance for experiment visualization
resource "google_vertex_ai_tensorboard" "experiments" {
  provider = google-beta

  display_name = "trading-signals-experiments"
  description  = "Tensorboard for tracking GNN training experiments"
  region       = var.region
}

# Vertex AI Dataset for training data (optional, for managed training)
resource "google_vertex_ai_dataset" "trading_signals" {
  provider = google

  display_name = "trading-signals-dataset"
  region       = var.region

  metadata_schema_uri = "gs://google-cloud-aiplatform/schema/dataset/metadata/tabular_1.0.0.yaml"
}

# Model Registry endpoint for serving predictions
resource "google_vertex_ai_endpoint" "prediction" {
  provider = google

  name         = "trading-signals-endpoint"
  display_name = "Trading Signals Prediction Endpoint"
  description  = "Endpoint for serving GNN trading predictions"
  location     = var.region
  region       = var.region
}