"""
BigQuery-based progress tracking for historical backfills.
"""

from datetime import datetime
from typing import Optional, List, Dict, Tuple
import pandas as pd
from loguru import logger

from src.utils.bigquery.client import BigQueryClient


class BackfillTracker:
    """Tracks backfill progress in BigQuery to enable resumable operations."""
    
    def __init__(self, bq_client: BigQueryClient):
        self.bq_client = bq_client
        self.progress_table = "backfill_progress"
        self.checkpoint_table = "backfill_checkpoints"
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Create progress tracking tables if they don't exist."""
        # Progress table - tracks overall backfill status
        progress_schema = [
            {"name": "backfill_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "start_year", "type": "INTEGER", "mode": "REQUIRED"},
            {"name": "end_year", "type": "INTEGER", "mode": "REQUIRED"},
            {"name": "current_year", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "current_month", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "status", "type": "STRING", "mode": "REQUIRED"},  # pending, in_progress, completed, failed
            {"name": "data_types", "type": "STRING", "mode": "REPEATED"},
            {"name": "total_months", "type": "INTEGER", "mode": "REQUIRED"},
            {"name": "completed_months", "type": "INTEGER", "mode": "REQUIRED"},
            {"name": "started_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "updated_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "completed_at", "type": "TIMESTAMP", "mode": "NULLABLE"},
            {"name": "error_message", "type": "STRING", "mode": "NULLABLE"},
            {"name": "inserted_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
        ]
        
        # Checkpoint table - tracks individual year-month completions
        checkpoint_schema = [
            {"name": "backfill_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "symbol", "type": "STRING", "mode": "REQUIRED"},
            {"name": "year", "type": "INTEGER", "mode": "REQUIRED"},
            {"name": "month", "type": "INTEGER", "mode": "REQUIRED"},
            {"name": "data_type", "type": "STRING", "mode": "REQUIRED"},
            {"name": "status", "type": "STRING", "mode": "REQUIRED"},  # completed, failed
            {"name": "records_fetched", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "completed_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "inserted_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "error_message", "type": "STRING", "mode": "NULLABLE"},
        ]
        
        # Convert schema to BigQuery SchemaField objects
        from google.cloud import bigquery
        
        progress_fields = [
            bigquery.SchemaField(field["name"], field["type"], mode=field["mode"])
            for field in progress_schema
        ]
        
        checkpoint_fields = [
            bigquery.SchemaField(field["name"], field["type"], mode=field["mode"])
            for field in checkpoint_schema
        ]
        
        # Create tables using the existing create_table method
        try:
            self.bq_client.create_table(self.progress_table, progress_fields)
        except Exception as e:
            if "already exists" not in str(e):
                raise
                
        try:
            self.bq_client.create_table(self.checkpoint_table, checkpoint_fields)
        except Exception as e:
            if "already exists" not in str(e):
                raise
    
    def start_backfill(
        self,
        backfill_id: str,
        start_year: int,
        end_year: int,
        data_types: List[str]
    ) -> Dict:
        """Start a new backfill job and track it."""
        # Calculate total months
        total_months = (end_year - start_year + 1) * 12
        
        # Check if this backfill already exists
        existing = self.get_backfill_status(backfill_id)
        if existing and existing['status'] == 'completed':
            logger.warning(f"Backfill {backfill_id} already completed")
            return existing
        
        # Create or update progress record
        progress_data = pd.DataFrame([{
            'backfill_id': backfill_id,
            'start_year': start_year,
            'end_year': end_year,
            'current_year': end_year,  # Start from most recent
            'current_month': 12,
            'status': 'in_progress',
            'data_types': data_types,
            'total_months': total_months,
            'completed_months': 0,
            'started_at': pd.Timestamp.now(),
            'updated_at': pd.Timestamp.now(),
            'completed_at': None,
            'error_message': None
        }])
        
        # Insert the progress data
        self.bq_client.insert_dataframe(
            progress_data,
            self.progress_table,
            if_exists='merge',
            merge_keys=['backfill_id']
        )
        
        logger.info(f"Started backfill {backfill_id}: {start_year}-{end_year}")
        return progress_data.iloc[0].to_dict()
    
    def get_backfill_status(self, backfill_id: str) -> Optional[Dict]:
        """Get current status of a backfill."""
        query = f"""
        SELECT *
        FROM {self.bq_client.dataset_id}.{self.progress_table}
        WHERE backfill_id = @backfill_id
        """
        
        from google.cloud import bigquery
        params = [
            bigquery.ScalarQueryParameter('backfill_id', 'STRING', backfill_id)
        ]
        result = self.bq_client.query(query, params)
        
        if result.empty:
            return None
        
        return result.iloc[0].to_dict()
    
    def get_next_month_to_process(
        self,
        backfill_id: str
    ) -> Optional[Tuple[int, int]]:
        """Get the next year-month to process, moving backward in time."""
        status = self.get_backfill_status(backfill_id)
        if not status or status['status'] != 'in_progress':
            return None
        
        current_year = status['current_year']
        current_month = status['current_month']
        
        # Move backward
        if current_month > 1:
            return (current_year, current_month)
        elif current_year > status['start_year']:
            return (current_year - 1, 12)
        else:
            # We've reached the start
            return None
    
    def checkpoint_month(
        self,
        backfill_id: str,
        symbol: str,
        year: int,
        month: int,
        data_type: str,
        records_fetched: int,
        error: Optional[str] = None
    ):
        """Record completion of a specific symbol-month."""
        checkpoint_data = pd.DataFrame([{
            'backfill_id': backfill_id,
            'symbol': symbol,
            'year': year,
            'month': month,
            'data_type': data_type,
            'status': 'failed' if error else 'completed',
            'records_fetched': records_fetched if not error else None,
            'completed_at': datetime.now(),
            'error_message': error
        }])
        
        self.bq_client.insert_dataframe(
            checkpoint_data,
            self.checkpoint_table
        )
    
    def update_progress(
        self,
        backfill_id: str,
        year: int,
        month: int,
        increment_completed: bool = True
    ):
        """Update current progress position."""
        status = self.get_backfill_status(backfill_id)
        if not status:
            return
        
        # Calculate next position (moving backward)
        next_month = month - 1
        next_year = year
        
        if next_month < 1:
            next_month = 12
            next_year = year - 1
        
        # Check if we're done
        is_complete = (year <= status['start_year'] and month == 1)
        
        updates = {
            'current_year': next_year if not is_complete else year,
            'current_month': next_month if not is_complete else month,
            'status': 'completed' if is_complete else 'in_progress',
            'updated_at': datetime.now()
        }
        
        if increment_completed:
            updates['completed_months'] = status['completed_months'] + 1
        
        if is_complete:
            updates['completed_at'] = datetime.now()
        
        # Update the record
        self._update_progress_record(backfill_id, updates)
    
    def _update_progress_record(self, backfill_id: str, updates: Dict):
        """Update specific fields in the progress record."""
        set_clause = ', '.join([f"{k} = @{k}" for k in updates.keys()])
        
        query = f"""
        UPDATE {self.bq_client.dataset_id}.{self.progress_table}
        SET {set_clause}
        WHERE backfill_id = @backfill_id
        """
        
        params = {'backfill_id': backfill_id}
        params.update(updates)
        
        from google.cloud import bigquery
        query_params = []
        for key, value in params.items():
            if isinstance(value, datetime):
                query_params.append(bigquery.ScalarQueryParameter(key, 'TIMESTAMP', value))
            elif isinstance(value, int):
                query_params.append(bigquery.ScalarQueryParameter(key, 'INT64', value))
            else:
                query_params.append(bigquery.ScalarQueryParameter(key, 'STRING', str(value)))
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        self.bq_client.client.query(query, job_config=job_config).result()
    
    def get_completed_months(
        self,
        backfill_id: str,
        symbol: str,
        data_type: str
    ) -> List[Tuple[int, int]]:
        """Get list of completed year-months for a symbol."""
        query = f"""
        SELECT DISTINCT year, month
        FROM {self.bq_client.dataset_id}.{self.checkpoint_table}
        WHERE backfill_id = @backfill_id
          AND symbol = @symbol
          AND data_type = @data_type
          AND status = 'completed'
        ORDER BY year DESC, month DESC
        """
        
        from google.cloud import bigquery
        params = [
            bigquery.ScalarQueryParameter('backfill_id', 'STRING', backfill_id),
            bigquery.ScalarQueryParameter('symbol', 'STRING', symbol),
            bigquery.ScalarQueryParameter('data_type', 'STRING', data_type)
        ]
        result = self.bq_client.query(query, params)
        
        return [(row['year'], row['month']) for _, row in result.iterrows()]
    
    def get_active_backfills(self) -> pd.DataFrame:
        """Get all active backfill jobs."""
        query = f"""
        SELECT *
        FROM {self.bq_client.dataset_id}.{self.progress_table}
        WHERE status = 'in_progress'
        ORDER BY started_at DESC
        """
        
        return self.bq_client.query(query)
    
    def fail_backfill(self, backfill_id: str, error_message: str):
        """Mark a backfill as failed."""
        self._update_progress_record(
            backfill_id,
            {
                'status': 'failed',
                'error_message': error_message,
                'updated_at': datetime.now()
            }
        )