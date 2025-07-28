"""
Core BigQuery operations for data manipulation.
"""
from typing import List, Optional, Dict, Any
import pandas as pd
from google.cloud import bigquery
from loguru import logger

from config.settings import BQ_TABLES, INGESTION_CONFIG
from .base import BigQueryBase
from .utils import retry_on_error


class BigQueryOperations(BigQueryBase):
    """BigQuery operations for data insertion and querying."""

    @retry_on_error()
    def insert_dataframe(
            self,
            df: pd.DataFrame,
            table_name: str,
            chunk_size: Optional[int] = None,
            if_exists: str = 'append',
            merge_keys: Optional[List[str]] = None
    ) -> None:
        """
        Insert DataFrame to BigQuery table with support for merge operations.

        Args:
            df: DataFrame to insert
            table_name: Target table name
            chunk_size: Number of rows per chunk
            if_exists: How to behave if table exists ('append', 'replace', or 'merge')
            merge_keys: List of columns to use as merge keys (for deduplication)
        """
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")
        chunk_size = chunk_size or INGESTION_CONFIG['chunk_size']

        # Add insertion timestamp if not present
        if 'inserted_at' not in df.columns:
            df['inserted_at'] = pd.Timestamp.now()

        # If merge operation is requested
        if if_exists == 'merge' and merge_keys:
            # Use a temporary table for merge
            temp_table_id = f"{table_id}_temp_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

            # Load data to temporary table first
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
            )

            job = self.client.load_table_from_dataframe(
                df, temp_table_id, job_config=job_config
            )
            job.result()

            # Perform merge operation
            merge_keys_str = ', '.join([f't.{key} = s.{key}' for key in merge_keys])
            update_cols = [col for col in df.columns if col not in merge_keys]
            
            # Check table schema to identify TIMESTAMP columns
            table = self.client.get_table(table_id)
            timestamp_cols = {field.name for field in table.schema if field.field_type == 'TIMESTAMP'}
            
            # Build update string with TIMESTAMP conversion for timestamp columns
            update_parts = []
            for col in update_cols:
                if col in timestamp_cols:
                    update_parts.append(f't.{col} = TIMESTAMP(s.{col})')
                else:
                    update_parts.append(f't.{col} = s.{col}')
            update_str = ', '.join(update_parts)

            merge_query = f"""
            MERGE `{table_id}` t
            USING `{temp_table_id}` s
            ON {merge_keys_str}
            WHEN MATCHED THEN
                UPDATE SET {update_str}
            WHEN NOT MATCHED THEN
                INSERT ({', '.join(df.columns)})
                VALUES ({', '.join([f'TIMESTAMP(s.{col})' if col in timestamp_cols else f's.{col}' for col in df.columns])})
            """

            self.client.query(merge_query).result()

            # Drop temporary table
            self.client.delete_table(temp_table_id)

            logger.info(f"Merged {len(df)} rows into {table_id}")

        else:
            # Standard insert operation
            total_rows = len(df)
            rows_inserted = 0

            for i in range(0, total_rows, chunk_size):
                chunk = df.iloc[i:i + chunk_size]

                job_config = bigquery.LoadJobConfig(
                    write_disposition=(
                        bigquery.WriteDisposition.WRITE_APPEND
                        if if_exists == 'append'
                        else bigquery.WriteDisposition.WRITE_TRUNCATE
                    )
                )

                job = self.client.load_table_from_dataframe(
                    chunk, table_id, job_config=job_config
                )
                job.result()  # Wait for job to complete

                rows_inserted += len(chunk)
                logger.info(f"Inserted {rows_inserted}/{total_rows} rows to {table_id}")

    @retry_on_error()
    def query(
            self,
            query: str,
            params: Optional[List[bigquery.ScalarQueryParameter]] = None,
            use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame.

        Args:
            query: SQL query to execute
            params: Query parameters
            use_cache: Whether to use query cache

        Returns:
            Query results as DataFrame
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=params or [],
            use_query_cache=use_cache
        )

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        return results.to_dataframe()

    def get_latest_date(self, table_name: str, ticker: Optional[str] = None) -> Optional[pd.Timestamp]:
        """
        Get the latest date for a ticker in a table.

        Args:
            table_name: Table name
            ticker: Optional ticker to filter by

        Returns:
            Latest date or None if no data
        """
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        query = f"""
        SELECT MAX(date) as latest_date
        FROM `{table_id}`
        """

        if ticker:
            query += f" WHERE ticker = '{ticker}'"

        try:
            result = self.query(query)
            if not result.empty and result['latest_date'].iloc[0] is not None:
                return pd.Timestamp(result['latest_date'].iloc[0])
        except Exception as e:
            logger.warning(f"Could not get latest date: {e}")

        return None

    def batch_insert(
            self,
            records: List[Dict[str, Any]],
            table_name: str,
            batch_size: int = 1000
    ) -> int:
        """
        Insert records in batches.

        Args:
            records: List of records to insert
            table_name: Target table name
            batch_size: Records per batch

        Returns:
            Number of records inserted
        """
        if not records:
            return 0

        df = pd.DataFrame(records)
        self.insert_dataframe(df, table_name, chunk_size=batch_size)
        return len(records)

    def stream_insert(
            self,
            rows: List[Dict[str, Any]],
            table_name: str
    ) -> List[Dict[str, Any]]:
        """
        Stream insert rows to BigQuery.

        Args:
            rows: List of rows to insert
            table_name: Target table name

        Returns:
            List of errors if any
        """
        table_ref = self.get_table_reference(table_name)
        table = self.client.get_table(table_ref)

        errors = self.client.insert_rows_json(table, rows)

        if errors:
            logger.error(f"Stream insert errors: {errors}")
        else:
            logger.info(f"Streamed {len(rows)} rows to {table_name}")

        return errors

    def delete_data(
            self,
            table_name: str,
            where_clause: str
    ) -> int:
        """
        Delete data from table based on where clause.

        Args:
            table_name: Table name
            where_clause: WHERE clause for deletion

        Returns:
            Number of rows deleted
        """
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        query = f"""
        DELETE FROM `{table_id}`
        WHERE {where_clause}
        """

        query_job = self.client.query(query)
        query_job.result()

        rows_deleted = query_job.num_dml_affected_rows
        logger.info(f"Deleted {rows_deleted} rows from {table_name}")

        return rows_deleted