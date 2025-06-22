"""
REST API service for predictions and signals.
"""
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, date
import pandas as pd
from loguru import logger
import os
import numpy as np  # Add this line

from src.training.prediction_pipeline import PredictionPipeline, RealTimePredictionService
from src.utils.bigquery import BigQueryClient
from config.settings import BQ_TABLES


# Pydantic models
class PredictionRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers")
    prediction_date: Optional[str] = Field(None, description="Date for prediction (YYYY-MM-DD)")
    include_confidence: bool = Field(True, description="Include confidence scores")


class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: date
    predictions: Dict[str, float]
    confidence_scores: Optional[Dict[str, float]]
    model_version: str


class SignalRequest(BaseModel):
    tickers: List[str]
    horizon: str = Field("7d", description="Prediction horizon (1d, 7d, 30d, 60d)")
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    return_threshold: float = Field(0.02, ge=0.0)
    max_signals: int = Field(20, ge=1, le=100)


class SignalResponse(BaseModel):
    ticker: str
    signal: str
    horizon: str
    predicted_return: float
    confidence: float
    rank: int


class PortfolioRequest(BaseModel):
    signals: List[SignalResponse]
    max_positions: int = Field(20, ge=1, le=50)
    risk_parity: bool = Field(True)


class PortfolioAllocation(BaseModel):
    ticker: str
    weight: float
    signal: str
    predicted_return: float


class PerformanceMetrics(BaseModel):
    horizon: str
    direction_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    information_coefficient: float
    last_updated: datetime


# Initialize FastAPI app
app = FastAPI(
    title="Trading Signal API",
    description="API for GNN-based trading signal generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
prediction_service = None
bq_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global prediction_service, bq_client

    # Load model
    model_path = os.getenv("MODEL_PATH", "models/production/latest.pth")
    prediction_service = RealTimePredictionService(model_version=model_path)

    # Initialize BigQuery client
    bq_client = BigQueryClient()

    logger.info("API service initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": prediction_service is not None
    }


@app.post("/predict", response_model=List[PredictionResponse])
async def generate_predictions(request: PredictionRequest):
    """Generate predictions for specified tickers."""
    try:
        # Use current date if not specified
        prediction_date = request.prediction_date or datetime.now().strftime('%Y-%m-%d')

        # Generate predictions
        predictions_df = prediction_service.pipeline.generate_predictions(
            prediction_date=prediction_date,
            tickers=request.tickers
        )

        if predictions_df.empty:
            raise HTTPException(status_code=404, detail="No predictions generated")

        # Convert to response format
        responses = []
        for _, row in predictions_df.iterrows():
            pred_dict = {
                f"{h}d": row[f'horizon_{h}d']
                for h in [1, 7, 30, 60]
                if f'horizon_{h}d' in row
            }

            conf_dict = None
            if request.include_confidence:
                conf_dict = {
                    f"{h}d": row[f'confidence_{h}d']
                    for h in [1, 7, 30, 60]
                    if f'confidence_{h}d' in row
                }

            responses.append(PredictionResponse(
                ticker=row['ticker'],
                prediction_date=row['prediction_date'],
                predictions=pred_dict,
                confidence_scores=conf_dict,
                model_version=row.get('model_version', 'unknown')
            ))

        return responses

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals", response_model=List[SignalResponse])
async def generate_signals(request: SignalRequest):
    """Generate trading signals based on predictions."""
    try:
        # Get predictions
        predictions_df = prediction_service.get_latest_predictions(
            tickers=request.tickers,
            force_update=False
        )

        if predictions_df.empty:
            raise HTTPException(status_code=404, detail="No predictions available")

        # Generate signals
        signals_df = prediction_service.pipeline.generate_trading_signals(
            predictions_df,
            confidence_threshold=request.confidence_threshold,
            return_threshold=request.return_threshold
        )

        # Filter by horizon
        if request.horizon != "all":
            signals_df = signals_df[signals_df['horizon'] == request.horizon]

        # Limit number of signals
        signals_df = signals_df.nsmallest(request.max_signals, 'rank')

        # Convert to response format
        responses = []
        for _, signal in signals_df.iterrows():
            responses.append(SignalResponse(
                ticker=signal['ticker'],
                signal=signal['signal'],
                horizon=signal['horizon'],
                predicted_return=signal['predicted_return'],
                confidence=signal['confidence'],
                rank=int(signal['rank'])
            ))

        return responses

    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio", response_model=List[PortfolioAllocation])
async def create_portfolio(request: PortfolioRequest):
    """Create portfolio allocation from signals."""
    try:
        # Convert signals to DataFrame
        signals_data = [
            {
                'ticker': s.ticker,
                'signal': s.signal,
                'horizon': s.horizon,
                'predicted_return': s.predicted_return,
                'confidence': s.confidence,
                'strength': abs(s.predicted_return) * s.confidence,
                'rank': s.rank,
                'date': datetime.now().date()
            }
            for s in request.signals
        ]

        signals_df = pd.DataFrame(signals_data)

        # Create portfolio allocation
        portfolio_df = prediction_service.pipeline.create_portfolio_allocation(
            signals_df,
            max_positions=request.max_positions,
            risk_parity=request.risk_parity
        )

        # Convert to response format
        allocations = []
        for _, row in portfolio_df.iterrows():
            allocations.append(PortfolioAllocation(
                ticker=row['ticker'],
                weight=row['weight'],
                signal=row['signal'],
                predicted_return=row['predicted_return']
            ))

        return allocations

    except Exception as e:
        logger.error(f"Portfolio creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/{horizon}", response_model=PerformanceMetrics)
async def get_performance_metrics(
        horizon: str = Query(..., regex="^(1d|7d|30d|60d)$"),
        days: int = Query(30, ge=1, le=365)
):
    """Get model performance metrics for a specific horizon."""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

        # Query performance metrics
        query = f"""
        WITH predictions_actuals AS (
            SELECT 
                p.ticker,
                p.prediction_date,
                p.horizon_{horizon} as prediction,
                p.confidence_{horizon} as confidence,
                r.actual_{horizon} as actual
            FROM `{BQ_TABLES['predictions']}` p
            JOIN (
                SELECT 
                    ticker,
                    date,
                    (LEAD(close, {horizon[:-1]}) OVER (PARTITION BY ticker ORDER BY date) - close) / close as actual_{horizon}
                FROM `{BQ_TABLES['raw_ohlcv']}`
            ) r
            ON p.ticker = r.ticker AND p.prediction_date = r.date
            WHERE p.prediction_date BETWEEN '{start_date}' AND '{end_date}'
              AND r.actual_{horizon} IS NOT NULL
        )
        SELECT 
            COUNT(*) as total_predictions,
            AVG(CASE WHEN SIGN(prediction) = SIGN(actual) THEN 1 ELSE 0 END) as direction_accuracy,
            CORR(prediction, actual) as information_coefficient,
            AVG(prediction) as avg_prediction,
            AVG(actual) as avg_actual,
            STDDEV(actual) as actual_volatility
        FROM predictions_actuals
        """

        result = bq_client.query(query)

        if result.empty:
            raise HTTPException(status_code=404, detail="No performance data available")

        metrics = result.iloc[0]

        # Calculate Sharpe ratio (simplified)
        sharpe = metrics['avg_actual'] / metrics['actual_volatility'] * np.sqrt(252)

        return PerformanceMetrics(
            horizon=horizon,
            direction_accuracy=metrics['direction_accuracy'],
            sharpe_ratio=sharpe,
            max_drawdown=-0.15,  # Placeholder - implement proper calculation
            information_coefficient=metrics['information_coefficient'],
            last_updated=datetime.now()
        )

    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickers")
async def get_available_tickers():
    """Get list of available tickers."""
    try:
        query = f"""
        SELECT DISTINCT ticker, sector, name
        FROM `{BQ_TABLES['stock_metadata']}`
        ORDER BY sector, ticker
        """

        result = bq_client.query(query)

        return result.to_dict('records')

    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/history")
async def get_prediction_history(
        ticker: str,
        start_date: str = Query(..., regex="^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., regex="^\d{4}-\d{2}-\d{2}$")
):
    """Get historical predictions for a ticker."""
    try:
        query = f"""
        SELECT 
            prediction_date,
            horizon_1d,
            horizon_7d,
            horizon_30d,
            horizon_60d,
            confidence_1d,
            confidence_7d,
            confidence_30d,
            confidence_60d
        FROM `{BQ_TABLES['predictions']}`
        WHERE ticker = '{ticker}'
          AND prediction_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY prediction_date DESC
        """

        result = bq_client.query(query)

        if result.empty:
            raise HTTPException(status_code=404, detail="No historical predictions found")

        return result.to_dict('records')

    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
async def run_backtest(
        start_date: str = Query(..., regex="^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., regex="^\d{4}-\d{2}-\d{2}$"),
        initial_capital: float = Query(100000, ge=1000),
        position_size: float = Query(0.05, ge=0.01, le=0.2)
):
    """Run backtest on historical predictions."""
    try:
        # This would implement a full backtesting engine
        # For now, return a placeholder
        return {
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "final_value": initial_capital * 1.15,  # Placeholder
            "total_return": 0.15,
            "sharpe_ratio": 0.75,
            "max_drawdown": -0.12,
            "num_trades": 150,
            "win_rate": 0.55,
            "message": "Full backtesting implementation pending"
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)