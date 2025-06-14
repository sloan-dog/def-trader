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
- **Cloud Run**: Containerized services for data ingestion and predictions
- **Vertex AI**: Distributed model training
- **Cloud Scheduler**: Automated job orchestration

## Getting Started

### Prerequisites
- Google Cloud Platform account
- Python 3.9+
- Docker
- Terraform
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
3. Deploy infrastructure with Terraform
4. Set up Cloud Run services
5. Create scheduled jobs

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
├── docker/              # Containerization
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
- Vertex AI: ~$100-200 (training frequency)
- Cloud Storage: ~$10-20

Tips:
- Use BigQuery partitioning
- Schedule jobs during off-peak hours
- Clean up old model artifacts

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Alternative data sources (satellite, social media)
- [ ] Reinforcement learning for portfolio optimization
- [ ] Multi-asset class support
- [ ] Advanced risk models
- [ ] Interactive dashboards

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for educational and research purposes only. Always perform your own due diligence before making investment decisions. Past performance does not guarantee future results.