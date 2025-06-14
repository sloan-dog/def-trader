# GNN Trading Signal System Architecture

## System Overview

### Core Components

1. **Data Ingestion Layer**
    - Historical data fetcher (Alpha Vantage API)
    - Technical indicators calculator
    - Macro economic data collector
    - Sentiment analysis aggregator
    - Temporal feature generator

2. **Data Storage Layer**
    - Google BigQuery schema design
    - Partitioned tables for efficient querying
    - Data versioning strategy

3. **Feature Engineering Pipeline**
    - Graph construction from market relationships
    - Node features: stock-specific metrics
    - Edge features: sector relationships, correlations
    - Temporal encoding

4. **Model Architecture**
    - Temporal Graph Neural Network (T-GNN)
    - Multi-horizon prediction heads (1d, 7d, 30d, 60d)
    - Attention mechanisms for feature importance

5. **Training Pipeline**
    - Distributed training on Google Cloud
    - Model versioning and registry
    - Hyperparameter optimization
    - Backtesting framework

6. **Infrastructure**
    - Google Cloud Run for data ingestion jobs
    - Vertex AI for model training
    - Cloud Scheduler for orchestration
    - Terraform for IaC

## Data Schema

### BigQuery Tables

1. **raw_ohlcv**
    - ticker, date, open, high, low, close, volume, adjusted_close

2. **technical_indicators**
    - ticker, date, rsi, ema_9, ema_20, ema_50, vwap, macd, bb_upper, bb_lower, atr, sma, adx, obv

3. **macro_indicators**
    - date, gdp, cpi, pce, nfp, unemployment_rate, fed_funds_rate, yield_curve_spread, retail_sales, ism_mfg, ism_services, consumer_confidence, wti_crude, brent_crude, china_pmi, china_gdp, m2_money_supply

4. **sentiment_data**
    - ticker, date, sector, sentiment_score, volume_mentions, source

5. **temporal_features**
    - date, day_of_week, month, quarter, is_holiday, days_to_next_holiday, is_earnings_season

6. **stock_metadata**
    - ticker, sector, industry, market_cap_category, exchange

## Selected Stocks by Sector

### Technology
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- NVDA (Nvidia)

### Finance
- JPM (JPMorgan Chase)
- BAC (Bank of America)
- GS (Goldman Sachs)
- BRK.B (Berkshire Hathaway)

### Healthcare
- JNJ (Johnson & Johnson)
- UNH (UnitedHealth)
- PFE (Pfizer)
- ABBV (AbbVie)

### Consumer Discretionary
- AMZN (Amazon)
- TSLA (Tesla)
- HD (Home Depot)
- MCD (McDonald's)

### Energy
- XOM (Exxon Mobil)
- CVX (Chevron)
- COP (ConocoPhillips)
- SLB (Schlumberger)

### Industrials
- CAT (Caterpillar)
- BA (Boeing)
- UPS (United Parcel Service)
- HON (Honeywell)

### Consumer Staples
- PG (Procter & Gamble)
- KO (Coca-Cola)
- WMT (Walmart)
- PEP (PepsiCo)

### Utilities
- NEE (NextEra Energy)
- DUK (Duke Energy)
- SO (Southern Company)
- D (Dominion Energy)

### Materials
- LIN (Linde)
- APD (Air Products)
- ECL (Ecolab)
- NEM (Newmont)

### Real Estate
- PLD (Prologis)
- AMT (American Tower)
- CCI (Crown Castle)
- EQIX (Equinix)

## Project Structure

```
trading-signal-system/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── modules/
│       ├── bigquery/
│       ├── cloud_run/
│       ├── vertex_ai/
│       └── cloud_scheduler/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── stocks_config.yaml
│   └── indicators_config.yaml
├── src/
│   ├── __init__.py
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── alpha_vantage_client.py
│   │   ├── ohlcv_fetcher.py
│   │   ├── macro_data_fetcher.py
│   │   ├── sentiment_fetcher.py
│   │   └── data_validator.py
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── technical_indicators.py
│   │   ├── temporal_features.py
│   │   ├── graph_constructor.py
│   │   └── feature_normalizer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_architecture.py
│   │   ├── temporal_gnn.py
│   │   ├── attention_layers.py
│   │   └── prediction_heads.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   ├── loss_functions.py
│   │   ├── metrics.py
│   │   └── model_registry.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── bigquery_client.py
│   │   ├── logging_config.py
│   │   ├── decorators.py
│   │   └── date_utils.py
│   └── jobs/
│       ├── __init__.py
│       ├── daily_ingestion.py
│       ├── indicator_calculation.py
│       ├── training_job.py
│       └── backfill_job.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── backtest_results.ipynb
└── scripts/
    ├── deploy.sh
    ├── run_backfill.sh
    └── start_training.sh
```

## Implementation Timeline

### Phase 1: Data Infrastructure (Week 1-2)
- Set up GCP project and BigQuery schemas
- Implement Alpha Vantage data fetchers
- Create technical indicator calculators
- Deploy initial Cloud Run jobs

### Phase 2: Feature Engineering (Week 3-4)
- Build graph construction logic
- Implement temporal feature extraction
- Create feature normalization pipeline
- Set up data validation

### Phase 3: Model Development (Week 5-6)
- Design T-GNN architecture
- Implement multi-horizon prediction
- Create training pipeline
- Set up model registry

### Phase 4: Production Deployment (Week 7-8)
- Terraform infrastructure setup
- CI/CD pipeline configuration
- Monitoring and alerting
- Documentation and testing

## Next Steps

1. Set up GCP project and enable required APIs
2. Configure Alpha Vantage API access
3. Create BigQuery dataset and initial schemas
4. Begin implementing data ingestion modules