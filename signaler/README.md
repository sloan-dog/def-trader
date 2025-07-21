# GNN-Based Trading Signal System

A sophisticated trading signal generation system that uses Graph Neural Networks (GNN) to analyze market relationships and predict multi-horizon price movements.

## Overview

This system combines:
- **Temporal Graph Neural Networks** for capturing cross-stock relationships and temporal patterns
- **Comprehensive data ingestion** from multiple sources (OHLCV, technical indicators, macro data, sentiment)
- **Multi-horizon predictions** (1, 7, 30, 60 days)
- **Cloud-native architecture** on Google Cloud Platform
- **Automated training and prediction pipelines**

## Architecture

### Data Flow
1. **Data Ingestion**: Daily collection of market data, technical indicators, and macro indicators
2. **Feature Engineering**: Calculation of 20+ technical indicators and temporal features
3. **Graph Construction**: Dynamic relationship modeling between stocks
4. **Model Training**: Temporal GNN with attention mechanisms
5. **Prediction Generation**: Multi-horizon return predictions with confidence scores
6. **Signal Generation**: Actionable trading signals based on predictions

### Key Components

#### Data Sources
- **Market Data**: OHLCV data from Alpha Vantage
- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- **Macro Indicators**: GDP, CPI, unemployment, yield curves
- **Sentiment Data**: News sentiment analysis
- **Temporal Features**: Holidays, earnings seasons, market regimes

#### Model Architecture
- **Temporal GNN**: Combines graph convolutions with LSTM and attention
- **Multi-horizon heads**: Separate prediction heads for each time horizon
- **Confidence estimation**: Calibrated confidence scores for each prediction

#### Infrastructure
- **BigQuery**: Scalable data warehouse for all historical data
- **Cloud Run**: Containerized services for predictions and backfill
- **Cloud Run Jobs**: Batch processing for data ingestion (more efficient than services)
- **Vertex AI**: Distributed model training
- **Cloud Scheduler**: Automated job orchestration

## Getting Started

### Prerequisites
- Google Cloud Platform account
- Python 3.9+
- Docker
- Open Tofu (Open source terraform)
- Alpha Vantage API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourorg/trading-signal-system.git
cd trading-signal-system
```

2. Set up Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Set up GCP credentials:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Deployment

Deploy the entire system with:
```bash
./scripts/deploy.sh --project-id YOUR_PROJECT_ID --region us-central1
```

This will:
1. Enable required GCP APIs
2. Build and push Docker images
3. Deploy infrastructure with Terraform / Open Tofu
4. Set up Cloud Run services and jobs
5. Create scheduled jobs

#### Cloud Run Jobs for Batch Processing

Both the daily ingestion and backfill jobs are deployed as **Cloud Run Jobs** instead of services, which is more appropriate for batch processing:

**Benefits:**
- **No HTTP overhead** - Direct job execution
- **Better resource utilization** - Can use more CPU/memory efficiently
- **Proper retry logic** - Built-in retry mechanisms
- **Cost effective** - Only pay for actual execution time
- **Simpler monitoring** - Job-specific metrics and logs

**Deployment:**
```bash
# Deploy the ingestion job
./scripts/deploy_ingestion_job.sh YOUR_PROJECT_ID us-central1

# Deploy the backfill job
./scripts/deploy_backfill_job.sh YOUR_PROJECT_ID us-central1

# Execute jobs manually for testing
gcloud run jobs execute daily-ingestion-job --region=us-central1
gcloud run jobs execute backfill-job --region=us-central1 --args="--type=hourly"

# View job executions
gcloud run jobs executions list --job=daily-ingestion-job --region=us-central1
gcloud run jobs executions list --job=backfill-job --region=us-central1
```

**Backfill Job Types:**
- **Hourly**: Last 2 days of OHLCV data
- **Daily**: Last 7 days of OHLCV and macro data
- **Weekly**: Last 30 days of all data types
- **Historical**: Years of historical data with progress tracking

### Running Locally

#### Data Backfill
```bash
python -m src.jobs.backfill_job \
  --start-date 2022-01-01 \
  --end-date 2023-12-31 \
  --data-types all
```

#### Daily Ingestion
```bash
python -m src.jobs.daily_ingestion --date 2024-01-15
```

#### Model Training
```bash
python -m src.jobs.training_job \
  --end-date 2023-12-31 \
  --lookback-months 24
```

#### Generate Predictions
```bash
python -m src.training.prediction_pipeline \
  --model-version latest \
  --prediction-date 2024-01-15
```

## Configuration

### Stock Universe
The system tracks 40 carefully selected stocks across 10 sectors:
- Technology (AAPL, MSFT, GOOGL, NVDA)
- Financials (JPM, BAC, GS, BRK.B)
- Healthcare (JNJ, UNH, PFE, ABBV)
- And more...

Edit `config/stocks_config.yaml` to modify the stock universe.

### Model Parameters
Key parameters in `config/settings.py`:
- `historical_window`: 90 days of lookback
- `prediction_horizons`: [1, 7, 30, 60] days
- `hidden_dim`: 128 (GNN hidden dimension)
- `num_gnn_layers`: 3
- `attention_heads`: 4

### Technical Indicators
Configure indicators in `config/indicators_config.yaml`:
- RSI (14-day)
- EMA (9, 20, 50)
- MACD (12, 26, 9)
- Bollinger Bands (20, 2)
- And more...

## Usage Examples

### Generating Trading Signals
```python
from src.training.prediction_pipeline import PredictionPipeline

# Initialize pipeline
pipeline = PredictionPipeline(model_path="models/best_model.pth")

# Generate predictions
predictions = pipeline.generate_predictions(
    prediction_date="2024-01-15",
    tickers=["AAPL", "GOOGL", "JPM"]
)

# Convert to trading signals
signals = pipeline.generate_trading_signals(
    predictions,
    confidence_threshold=0.6,
    return_threshold=0.02
)
```

### Portfolio Construction
```python
# Create portfolio allocation
portfolio = pipeline.create_portfolio_allocation(
    signals,
    max_positions=20,
    risk_parity=True
)
```

## Model Performance

The system tracks multiple performance metrics:
- **Direction Accuracy**: >52% (better than random)
- **Sharpe Ratio**: Target >0.5
- **Information Coefficient**: Correlation between predictions and returns
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown

## Monitoring

### BigQuery Dashboards
Monitor system health and performance:
- Data ingestion status
- Prediction accuracy over time
- Model performance metrics
- System errors and alerts

### Logging
Logs are stored in:
- Local: `logs/` directory
- Cloud: Google Cloud Logging

### Alerts
Configure alerts for:
- Data ingestion failures
- Model performance degradation
- System errors

## Development

### Project Structure
```
trading-signal-system/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── data_ingestion/  # Data fetching modules
│   ├── feature_engineering/  # Feature calculation
│   ├── models/          # GNN model architecture
│   ├── training/        # Training and prediction
│   ├── utils/           # Utilities
│   └── jobs/            # Orchestration jobs
├── terraform/           # Infrastructure as code
│   └── modules/
│       ├── cloud_run/   # Cloud Run services
│       ├── cloud_run_jobs/  # Cloud Run Jobs
│       └── cloud_scheduler/ # Scheduled jobs
├── docker/              # Containerization
│   ├── Dockerfile.ingestion.job  # Batch job container
│   └── Dockerfile.ingestion      # Service container (legacy)
├── tests/               # Unit and integration tests
├── notebooks/           # Analysis notebooks
└── scripts/             # Deployment scripts
```

### Testing
Run tests with:
```bash
pytest tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Best Practices

### Data Quality
- Daily validation checks on ingested data
- Outlier detection and handling
- Missing data imputation strategies

### Model Management
- Version control for models
- A/B testing for new models
- Regular retraining schedule

### Risk Management
- Position sizing based on confidence
- Diversification across sectors
- Maximum drawdown limits

## Troubleshooting

### Common Issues

1. **Alpha Vantage Rate Limits**
    - Solution: Implement exponential backoff
    - Consider premium API key

2. **BigQuery Quota Exceeded**
    - Solution: Optimize queries
    - Use partitioned tables

3. **Model Training OOM**
    - Solution: Reduce batch size
    - Use gradient accumulation

## Cost Optimization

Estimated monthly costs:
- BigQuery: ~$50-100 (depending on data volume)
- Cloud Run: ~$20-50 (based on usage)
- Cloud Run Jobs: ~$10-30 (batch processing, pay per execution)
- Vertex AI: ~$100-200 (training frequency)
- Cloud Storage: ~$10-20

Tips:
- Use BigQuery partitioning
- Schedule jobs during off-peak hours
- Clean up old model artifacts
- Cloud Run Jobs are more cost-effective for batch workloads

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Alternative data sources (satellite, social media)
- [ ] Reinforcement learning for portfolio optimization
- [ ] Multi-asset class support
- [ ] Advanced risk models
- [ ] Interactive dashboards

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Local Development Setup - Incremental Terraform Approach

This guide explains how to set up just the essential GCP resources for local development, with the ability to expand to full production infrastructure later without conflicts.

## Quick Start

```bash
# 1. Make scripts executable
chmod +x ../scripts/terraform_minimal.sh
chmod +x ../scripts/terraform_check.sh

# 2. Create minimal resources for local development
../scripts/terraform_minimal.sh

# 3. Create service account key
gcloud iam service-accounts keys create ~/trading-system-key.json \
  --iam-account=trading-system@trading-signals-420-69.iam.gserviceaccount.com

# 4. Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/trading-system-key.json

# 5. Test the setup
python ../test_setup.py
```

## What Gets Created

The minimal setup creates only what you need for local development:

### ✅ Created by `terraform_minimal.sh`:
- **BigQuery**: Dataset and all tables
- **Service Account**: `trading-system` with necessary permissions
- **Secret Manager**: Storage for Alpha Vantage API key
- **Storage Bucket**: For model artifacts (optional but useful)
- **APIs**: Enables BigQuery, Storage, and Secret Manager APIs

### ❌ NOT Created (saved for later):
- Cloud Run services
- Cloud Scheduler jobs
- Vertex AI resources
- Monitoring/Alerting
- Additional service accounts

## Workflow

### Phase 1: Local Development
```bash
# Create minimal resources
./scripts/terraform_minimal.sh

# Check what exists
./scripts/terraform_check.sh

# Start developing locally
python -m src.jobs.backfill_job --start-date 2024-01-01 --end-date 2024-01-31
```

### Phase 2: Full Deployment (Later)
```bash
# When ready to deploy everything
terraform apply -var-file=terraform.tfvars

# This will:
# - Skip resources that already exist (BigQuery, etc.)
# - Add Cloud Run, Scheduler, and other production resources
# - No conflicts! Terraform recognizes existing resources
```

## Why This Works

1. **Terraform State**: All resources are tracked in the same state file
2. **Idempotency**: Terraform won't recreate resources that already exist
3. **Incremental**: You can add resources gradually as needed
4. **Cost-Effective**: Only pay for what you use during development

## Checking Resource Status

```bash
# See what resources currently exist
terraform state list

# See what would be created with full apply
./scripts/terraform_check.sh

# Get more details about a specific resource
terraform state show module.bigquery
```

## Common Commands

```bash
# Refresh state (sync with actual GCP resources)
terraform refresh -var-file=terraform.tfvars

# Destroy only specific resources (careful!)
terraform destroy -target=google_storage_bucket.models -var-file=terraform.tfvars

# Import existing resources (if created outside Terraform)
terraform import google_bigquery_dataset.trading_signals trading-signals-420-69:trading_signals
```

## Troubleshooting

### "Resource already exists" error
```bash
# Import the existing resource into Terraform state
terraform import <resource_type>.<resource_name> <resource_id>
```

### State lock issues
```bash
# Force unlock (use carefully)
terraform force-unlock <lock_id>
```

### Verify BigQuery access
```bash
bq ls
bq show trading_signals
```

## Cost Optimization

During development, you're only paying for:
- BigQuery storage: ~$0.02/GB/month
- BigQuery queries: $5/TB processed
- Secret Manager: $0.06/secret/month
- Storage bucket: ~$0.02/GB/month

The expensive resources (Cloud Run, Vertex AI) are created later when needed.

## Disclaimer

This system is for educational and research purposes only. Always perform your own due diligence before making investment decisions. Past performance does not guarantee future results.modi
