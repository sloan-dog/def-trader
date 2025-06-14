# BigQuery module for Trading Signal System

# Dataset
resource "google_bigquery_dataset" "trading_signals" {
  dataset_id                  = var.dataset_id
  friendly_name               = "Trading Signals Dataset"
  description                 = "Dataset for trading signal system data"
  location                    = var.location
  default_table_expiration_ms = var.default_table_expiration_ms

  labels = var.labels

  access {
    role          = "OWNER"
    user_by_email = google_service_account.bigquery_admin.email
  }

  access {
    role          = "READER"
    special_group = "projectReaders"
  }
}

# Service account for BigQuery operations
resource "google_service_account" "bigquery_admin" {
  account_id   = "bigquery-admin"
  display_name = "BigQuery Admin"
  description  = "Service account for BigQuery administrative tasks"
}

# Tables
resource "google_bigquery_table" "raw_ohlcv" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "raw_ohlcv"

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  clustering = ["ticker"]

  schema = jsonencode([
    {
      name = "ticker"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "open"
      type = "FLOAT64"
      mode = "REQUIRED"
    },
    {
      name = "high"
      type = "FLOAT64"
      mode = "REQUIRED"
    },
    {
      name = "low"
      type = "FLOAT64"
      mode = "REQUIRED"
    },
    {
      name = "close"
      type = "FLOAT64"
      mode = "REQUIRED"
    },
    {
      name = "volume"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "adjusted_close"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "inserted_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "technical_indicators" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "technical_indicators"

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  clustering = ["ticker"]

  schema = jsonencode([
    {
      name = "ticker"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "rsi"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "ema_9"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "ema_20"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "ema_50"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "vwap"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "macd"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "macd_signal"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "macd_hist"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "bb_upper"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "bb_middle"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "bb_lower"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "atr"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "sma_20"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "sma_50"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "sma_200"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "adx"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "obv"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "inserted_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "macro_indicators" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "macro_indicators"

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  schema = jsonencode([
    {
      name = "date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "gdp"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "gdp_growth"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "cpi"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "cpi_yoy"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "pce"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "nfp"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "unemployment_rate"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "fed_funds_rate"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "yield_curve_spread"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "retail_sales"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "ism_manufacturing"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "ism_services"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "consumer_confidence"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "wti_crude"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "brent_crude"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "china_pmi"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "china_gdp"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "m2_money_supply"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "inserted_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "sentiment_data" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "sentiment_data"

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  clustering = ["ticker", "sector"]

  schema = jsonencode([
    {
      name = "ticker"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "sector"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "sentiment_score"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "volume_mentions"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "source"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "inserted_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "temporal_features" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "temporal_features"

  time_partitioning {
    type  = "DAY"
    field = "date"
  }

  schema = jsonencode([
    {
      name = "date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "day_of_week"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "month"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "quarter"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "year"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "is_holiday"
      type = "BOOL"
      mode = "REQUIRED"
    },
    {
      name = "days_to_next_holiday"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "is_earnings_season"
      type = "BOOL"
      mode = "REQUIRED"
    },
    {
      name = "is_month_start"
      type = "BOOL"
      mode = "REQUIRED"
    },
    {
      name = "is_month_end"
      type = "BOOL"
      mode = "REQUIRED"
    },
    {
      name = "is_quarter_start"
      type = "BOOL"
      mode = "REQUIRED"
    },
    {
      name = "is_quarter_end"
      type = "BOOL"
      mode = "REQUIRED"
    },
    {
      name = "inserted_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "stock_metadata" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "stock_metadata"

  schema = jsonencode([
    {
      name = "ticker"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "sector"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "industry"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "market_cap_category"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "exchange"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "updated_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "predictions" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "predictions"

  time_partitioning {
    type  = "DAY"
    field = "prediction_date"
  }

  clustering = ["ticker", "model_version"]

  schema = jsonencode([
    {
      name = "prediction_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "model_version"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "ticker"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "prediction_date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "horizon_1d"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "horizon_7d"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "horizon_30d"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "horizon_60d"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "confidence_1d"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "confidence_7d"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "confidence_30d"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "confidence_60d"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "model_metadata" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "model_metadata"

  schema = jsonencode([
    {
      name = "model_version"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "model_type"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "training_start_date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "training_end_date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "validation_metrics"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "hyperparameters"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "feature_importance"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "model_path"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    }
  ])
}

resource "google_bigquery_table" "job_logs" {
  dataset_id = google_bigquery_dataset.trading_signals.dataset_id
  table_id   = "job_logs"

  time_partitioning {
    type  = "DAY"
    field = "run_date"
  }

  schema = jsonencode([
    {
      name = "job_name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "run_date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "start_time"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "end_time"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "duration_hours"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "success"
      type = "BOOL"
      mode = "REQUIRED"
    },
    {
      name = "parameters"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "step_results"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "error"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])
}

# Outputs
output "dataset_id" {
  value = google_bigquery_dataset.trading_signals.dataset_id
}

output "dataset_location" {
  value = google_bigquery_dataset.trading_signals.location
}

output "table_ids" {
  value = {
    raw_ohlcv           = google_bigquery_table.raw_ohlcv.table_id
    technical_indicators = google_bigquery_table.technical_indicators.table_id
    macro_indicators    = google_bigquery_table.macro_indicators.table_id
    sentiment_data      = google_bigquery_table.sentiment_data.table_id
    temporal_features   = google_bigquery_table.temporal_features.table_id
    stock_metadata      = google_bigquery_table.stock_metadata.table_id
    predictions         = google_bigquery_table.predictions.table_id
    model_metadata      = google_bigquery_table.model_metadata.table_id
    job_logs           = google_bigquery_table.job_logs.table_id
  }
}