"""
Cloud Run Service wrapper for backfill job functionality.
This service provides HTTP endpoints to trigger backfill operations via Cloud Scheduler.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from src.jobs.backfill_job import BackfillJob
from src.jobs.historical_backfill_job import HistoricalBackfillJob
from src.shared_logging import setup_logging
from src.utils.bigquery.client import BigQueryClient
from src.utils.bigquery.backfill_tracker import BackfillTracker

setup_logging()

app = FastAPI(title="Backfill Service")


class BackfillRequest(BaseModel):
    """Request model for backfill operations."""
    
    start_date: Optional[str] = Field(
        None, 
        description="Start date in YYYY-MM-DD format. Defaults to 30 days ago."
    )
    end_date: Optional[str] = Field(
        None,
        description="End date in YYYY-MM-DD format. Defaults to today."
    )
    data_types: Optional[list[str]] = Field(
        default=["ohlcv"],
        description="Types of data to backfill: ohlcv, macro, or both"
    )
    batch_size: Optional[int] = Field(
        default=10,
        description="Number of tickers to process in parallel"
    )
    # For scheduled jobs, we might want a rolling window
    use_rolling_window: Optional[bool] = Field(
        default=False,
        description="Use a rolling window based on days_back"
    )
    days_back: Optional[int] = Field(
        default=7,
        description="Number of days to backfill when using rolling window"
    )


class BackfillResponse(BaseModel):
    """Response model for backfill operations."""
    
    status: str
    message: str
    job_details: dict


class HistoricalBackfillRequest(BaseModel):
    """Request model for historical backfill operations."""
    
    start_year: int = Field(..., description="Start year (e.g., 1995)")
    end_year: int = Field(..., description="End year (e.g., 2024)")
    data_types: Optional[list[str]] = Field(
        default=["ohlcv"],
        description="Types of data to backfill"
    )
    batch_size: Optional[int] = Field(
        default=10,
        description="Number of tickers to process in parallel"
    )
    backfill_id: Optional[str] = Field(
        None,
        description="Unique ID for resuming existing backfill"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "service": "backfill"}


@app.post("/backfill", response_model=BackfillResponse)
async def trigger_backfill(
    request: BackfillRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger a backfill operation.
    
    This endpoint can be called by Cloud Scheduler to run periodic backfills.
    The actual backfill runs in the background to avoid timeout issues.
    """
    try:
        # Determine date range
        if request.use_rolling_window:
            # For scheduled jobs, use a rolling window
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=request.days_back)
        else:
            # Use provided dates or defaults
            end_date = (
                datetime.strptime(request.end_date, "%Y-%m-%d").date()
                if request.end_date
                else datetime.now().date()
            )
            start_date = (
                datetime.strptime(request.start_date, "%Y-%m-%d").date()
                if request.start_date
                else end_date - timedelta(days=30)
            )
        
        # Validate date range
        if start_date > end_date:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        # Log the request
        logger.info(
            f"Received backfill request: {start_date} to {end_date}, "
            f"data_types={request.data_types}, batch_size={request.batch_size}"
        )
        
        # Create job configuration
        job_config = {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "data_types": request.data_types,
            "batch_size": request.batch_size,
        }
        
        # Add background task to run the backfill
        background_tasks.add_task(
            run_backfill_job,
            start_date=start_date,
            end_date=end_date,
            data_types=request.data_types,
            batch_size=request.batch_size
        )
        
        return BackfillResponse(
            status="accepted",
            message=f"Backfill job triggered for {start_date} to {end_date}",
            job_details=job_config
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger backfill: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger backfill: {str(e)}"
        )


async def run_backfill_job(
    start_date: datetime.date,
    end_date: datetime.date,
    data_types: list[str],
    batch_size: int
):
    """
    Run the backfill job asynchronously.
    
    This function is executed in the background to avoid request timeouts.
    """
    try:
        logger.info(f"Starting backfill job: {start_date} to {end_date}")
        
        # Initialize and run the backfill job
        job = BackfillJob(
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            batch_size=batch_size
        )
        
        # Run the job
        job.run()
        
        logger.info("Backfill job completed successfully")
        
    except Exception as e:
        logger.error(f"Backfill job failed: {str(e)}")
        # In a production system, you might want to send alerts here
        raise


@app.post("/backfill/daily")
async def trigger_daily_backfill(background_tasks: BackgroundTasks):
    """
    Endpoint specifically for daily scheduled backfills.
    Backfills the last 7 days of data.
    """
    request = BackfillRequest(
        use_rolling_window=True,
        days_back=7,
        data_types=["ohlcv"],
        batch_size=10
    )
    return await trigger_backfill(request, background_tasks)


@app.post("/backfill/hourly")
async def trigger_hourly_backfill(background_tasks: BackgroundTasks):
    """
    Endpoint specifically for hourly scheduled backfills.
    Backfills the last 2 days of data to ensure recent data is always up to date.
    """
    request = BackfillRequest(
        use_rolling_window=True,
        days_back=2,
        data_types=["ohlcv"],
        batch_size=20  # Higher batch size for smaller date range
    )
    return await trigger_backfill(request, background_tasks)


@app.post("/backfill/weekly")
async def trigger_weekly_backfill(background_tasks: BackgroundTasks):
    """
    Endpoint specifically for weekly scheduled backfills.
    Backfills the last 30 days of data.
    """
    request = BackfillRequest(
        use_rolling_window=True,
        days_back=30,
        data_types=["ohlcv", "macro"],
        batch_size=10
    )
    return await trigger_backfill(request, background_tasks)


@app.post("/backfill/historical", response_model=BackfillResponse)
async def trigger_historical_backfill(
    request: HistoricalBackfillRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger a historical backfill operation.
    
    This is designed for backfilling large date ranges (years of data).
    The job tracks progress and can be resumed if interrupted.
    """
    try:
        # Validate years
        current_year = datetime.now().year
        if request.start_year > request.end_year:
            raise HTTPException(400, "Start year must be before end year")
        if request.start_year < 1990:
            raise HTTPException(400, "Start year must be 1990 or later")
        if request.end_year > current_year:
            raise HTTPException(400, f"End year cannot be after {current_year}")
        
        # Create job configuration
        job_config = {
            "start_year": request.start_year,
            "end_year": request.end_year,
            "data_types": request.data_types,
            "batch_size": request.batch_size,
            "backfill_id": request.backfill_id or f"historical_{request.start_year}_{request.end_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        logger.info(f"Starting historical backfill: {job_config}")
        
        # Add background task to run the historical backfill
        background_tasks.add_task(
            run_historical_backfill,
            start_year=request.start_year,
            end_year=request.end_year,
            data_types=request.data_types,
            batch_size=request.batch_size,
            backfill_id=job_config["backfill_id"]
        )
        
        return BackfillResponse(
            status="accepted",
            message=f"Historical backfill triggered for {request.start_year}-{request.end_year}",
            job_details=job_config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger historical backfill: {str(e)}")
        raise HTTPException(500, f"Failed to trigger backfill: {str(e)}")


async def run_historical_backfill(
    start_year: int,
    end_year: int,
    data_types: list[str],
    batch_size: int,
    backfill_id: str
):
    """Run the historical backfill job asynchronously."""
    try:
        logger.info(f"Starting historical backfill job: {backfill_id}")
        
        job = HistoricalBackfillJob(
            start_year=start_year,
            end_year=end_year,
            data_types=data_types,
            batch_size=batch_size,
            backfill_id=backfill_id
        )
        
        job.run(resume=True)
        
        logger.info(f"Historical backfill job completed: {backfill_id}")
        
    except Exception as e:
        logger.error(f"Historical backfill job failed: {str(e)}")
        raise


@app.get("/backfill/status/{backfill_id}")
async def get_backfill_status(backfill_id: str):
    """Get the status of a historical backfill job."""
    try:
        bq_client = BigQueryClient()
        tracker = BackfillTracker(bq_client)
        
        status = tracker.get_backfill_status(backfill_id)
        if not status:
            raise HTTPException(404, f"Backfill {backfill_id} not found")
        
        # Calculate additional metrics
        if status['total_months'] > 0:
            progress_pct = (status['completed_months'] / status['total_months']) * 100
            status['progress_percentage'] = round(progress_pct, 2)
        
        # Estimate time remaining
        if status['status'] == 'in_progress' and status['completed_months'] > 0:
            elapsed = (datetime.utcnow() - status['started_at']).total_seconds()
            rate = status['completed_months'] / (elapsed / 3600)  # months per hour
            remaining_months = status['total_months'] - status['completed_months']
            estimated_hours = remaining_months / rate if rate > 0 else 0
            status['estimated_hours_remaining'] = round(estimated_hours, 1)
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backfill status: {str(e)}")
        raise HTTPException(500, f"Failed to get status: {str(e)}")


@app.get("/backfill/active")
async def get_active_backfills():
    """Get all active historical backfill jobs."""
    try:
        bq_client = BigQueryClient()
        tracker = BackfillTracker(bq_client)
        
        active_jobs = tracker.get_active_backfills()
        
        # Convert DataFrame to list of dicts
        jobs = active_jobs.to_dict('records')
        
        # Add progress percentage to each job
        for job in jobs:
            if job['total_months'] > 0:
                progress_pct = (job['completed_months'] / job['total_months']) * 100
                job['progress_percentage'] = round(progress_pct, 2)
        
        return {"active_backfills": jobs}
        
    except Exception as e:
        logger.error(f"Failed to get active backfills: {str(e)}")
        raise HTTPException(500, f"Failed to get active backfills: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)