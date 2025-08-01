"""
Parquet storage client for GCS operations.
Handles reading/writing partitioned Parquet files to Google Cloud Storage.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.fs import GcsFileSystem
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import json
from src.utils import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

class ParquetStorageClient:
    """Client for managing Parquet files on Google Cloud Storage."""
    
    def __init__(self, bucket_name: str, project_id: Optional[str] = None):
        """
        Initialize the Parquet storage client.
        
        Args:
            bucket_name: GCS bucket name (without gs:// prefix)
            project_id: GCP project ID (optional, uses default if not provided)
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        
        # Initialize GCS filesystem
        self.fs = GcsFileSystem(
            project_id=project_id,
            access_token=None  # Uses default credentials
        )
        
        # Base path for all operations
        self.base_path = f"{bucket_name}"
        
        logger.info(f"Initialized ParquetStorageClient for bucket: {bucket_name}")
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        path: str,
        partition_cols: Optional[List[str]] = None,
        compression: str = 'snappy',
        existing_data_behavior: str = 'overwrite_or_ignore'
    ) -> Dict[str, Any]:
        """
        Write DataFrame to Parquet file(s) in GCS.
        
        Args:
            df: DataFrame to write
            path: Path within bucket (e.g., 'ohlcv/raw')
            partition_cols: Columns to partition by
            compression: Compression type ('snappy', 'gzip', 'brotli', 'lz4', 'zstd')
            existing_data_behavior: How to handle existing data
                - 'overwrite_or_ignore': Default PyArrow behavior
                - 'error': Raise error if data exists
                - 'delete_matching': Delete matching partitions first
        
        Returns:
            Dict with write metadata
        """
        full_path = f"{self.base_path}/{path}"
        start_time = datetime.now()
        
        try:
            # Ensure datetime columns are properly formatted
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = pd.to_datetime(df[col])
            
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df)
            
            # Write to GCS
            pq.write_to_dataset(
                table,
                root_path=full_path,
                partition_cols=partition_cols,
                compression=compression,
                existing_data_behavior=existing_data_behavior,
                filesystem=self.fs
            )
            
            # Calculate metadata
            metadata = {
                'path': full_path,
                'rows_written': len(df),
                'partitions': partition_cols,
                'compression': compression,
                'write_time_seconds': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Wrote {len(df)} rows to {full_path}")
            return metadata
            
        except Exception as e:
            logger.error("Failed to write to {full_path}")
            raise
    
    def read_dataframe(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
        date_range: Optional[tuple] = None
    ) -> pd.DataFrame:
        """
        Read Parquet file(s) from GCS into DataFrame.
        
        Args:
            path: Path within bucket
            columns: Specific columns to read
            filters: PyArrow filters for partition pruning
                Example: [('symbol', '=', 'AAPL'), ('date', '>=', '2024-01-01')]
            date_range: Convenience filter for date columns (start_date, end_date)
        
        Returns:
            DataFrame with requested data
        """
        full_path = f"{self.base_path}/{path}"
        
        # Add date range to filters if provided
        if date_range and len(date_range) == 2:
            if filters is None:
                filters = []
            filters.extend([
                ('date', '>=', date_range[0]),
                ('date', '<=', date_range[1])
            ])
        
        try:
            # Read dataset
            dataset = pq.ParquetDataset(
                full_path,
                filesystem=self.fs,
                filters=filters
            )
            
            # Read to pandas
            df = dataset.read(columns=columns).to_pandas()
            
            logger.info(f"Read {len(df)} rows from {full_path}")
            return df
            
        except Exception as e:
            logger.error("Failed to read from {full_path}")
            raise
    
    def append_dataframe(
        self,
        df: pd.DataFrame,
        path: str,
        partition_cols: Optional[List[str]] = None,
        deduplicate_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Append DataFrame to existing Parquet dataset.
        
        Args:
            df: DataFrame to append
            path: Path within bucket
            partition_cols: Columns to partition by
            deduplicate_cols: Columns to use for deduplication
                If provided, will check for duplicates before appending
        
        Returns:
            Dict with append metadata
        """
        if deduplicate_cols:
            # Check for existing data
            try:
                existing_df = self.read_dataframe(path)
                
                # Create composite key for deduplication
                df['_dedup_key'] = df[deduplicate_cols].astype(str).agg('_'.join, axis=1)
                existing_df['_dedup_key'] = existing_df[deduplicate_cols].astype(str).agg('_'.join, axis=1)
                
                # Filter out duplicates
                new_records = df[~df['_dedup_key'].isin(existing_df['_dedup_key'])]
                new_records = new_records.drop('_dedup_key', axis=1)
                
                if len(new_records) == 0:
                    logger.info(f"No new records to append to {path}")
                    return {
                        'rows_appended': 0,
                        'duplicates_skipped': len(df)
                    }
                
                df = new_records
                
            except Exception as e:
                logger.info(f"No existing data found at {path}, proceeding with full append")
        
        # Append data
        metadata = self.write_dataframe(
            df=df,
            path=path,
            partition_cols=partition_cols,
            existing_data_behavior='overwrite_or_ignore'
        )
        
        metadata['operation'] = 'append'
        return metadata
    
    def list_partitions(
        self,
        path: str,
        partition_col: str
    ) -> List[str]:
        """
        List all partition values for a given partition column.
        
        Args:
            path: Path within bucket
            partition_col: Partition column name
        
        Returns:
            List of partition values
        """
        full_path = f"{self.base_path}/{path}"
        
        try:
            # List directories at path
            items = self.fs.ls(full_path)
            
            # Extract partition values
            partitions = []
            for item in items:
                if self.fs.isdir(item):
                    # Extract partition value from path
                    parts = item.split('/')
                    for part in parts:
                        if part.startswith(f"{partition_col}="):
                            value = part.split('=', 1)[1]
                            partitions.append(value)
            
            return sorted(list(set(partitions)))
            
        except Exception as e:
            logger.error("Failed to list partitions at {full_path}")
            return []
    
    def delete_partition(
        self,
        path: str,
        partition_filter: Dict[str, Any]
    ) -> bool:
        """
        Delete specific partition(s) based on filter.
        
        Args:
            path: Path within bucket
            partition_filter: Dict of partition column to value
                Example: {'symbol': 'AAPL', 'date': '2024-01-01'}
        
        Returns:
            Success boolean
        """
        full_path = f"{self.base_path}/{path}"
        
        # Build partition path
        partition_parts = []
        for col, value in partition_filter.items():
            partition_parts.append(f"{col}={value}")
        
        partition_path = f"{full_path}/{'/'.join(partition_parts)}"
        
        try:
            # Delete recursively
            self.fs.rm(partition_path, recursive=True)
            logger.info(f"Deleted partition: {partition_path}")
            return True
            
        except Exception as e:
            logger.error("Failed to delete partition {partition_path}")
            return False
    
    def get_dataset_info(self, path: str) -> Dict[str, Any]:
        """
        Get metadata about a Parquet dataset.
        
        Args:
            path: Path within bucket
        
        Returns:
            Dict with dataset information
        """
        full_path = f"{self.base_path}/{path}"
        
        try:
            dataset = pq.ParquetDataset(full_path, filesystem=self.fs)
            
            # Get schema
            schema = dataset.schema.to_arrow_schema()
            
            # Get partition info
            partitioning = dataset.partitioning
            
            # Count total rows (this reads metadata only)
            total_rows = sum(fragment.metadata.num_rows for fragment in dataset.fragments)
            
            # Get file count and size
            files = list(dataset.files)
            total_size = sum(
                self.fs.info(f"{self.bucket_name}/{file}").get('size', 0)
                for file in files
            )
            
            info = {
                'path': full_path,
                'total_rows': total_rows,
                'file_count': len(files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'schema': {field.name: str(field.type) for field in schema},
                'partitioning': str(partitioning) if partitioning else None
            }
            
            return info
            
        except Exception as e:
            logger.error("Failed to get dataset info for {full_path}")
            raise
    
    def read_latest_partition(
        self,
        path: str,
        partition_col: str = 'date',
        filters: Optional[List[tuple]] = None
    ) -> pd.DataFrame:
        """
        Read data from the latest partition.
        
        Args:
            path: Path within bucket
            partition_col: Column to determine latest by
            filters: Additional filters to apply
        
        Returns:
            DataFrame from latest partition
        """
        # Get all partitions
        partitions = self.list_partitions(path, partition_col)
        
        if not partitions:
            logger.warning(f"No partitions found at {path}")
            return pd.DataFrame()
        
        # Get latest partition
        latest = sorted(partitions)[-1]
        
        # Add to filters
        if filters is None:
            filters = []
        filters.append((partition_col, '=', latest))
        
        return self.read_dataframe(path, filters=filters)
    
    def parallel_write(
        self,
        dataframes: Dict[str, pd.DataFrame],
        base_path: str,
        partition_cols: Optional[List[str]] = None,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Write multiple DataFrames in parallel.
        
        Args:
            dataframes: Dict of {suffix: DataFrame}
            base_path: Base path for all writes
            partition_cols: Partition columns (same for all)
            max_workers: Number of parallel workers
        
        Returns:
            Dict with results for each write
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for suffix, df in dataframes.items():
                path = f"{base_path}/{suffix}"
                future = executor.submit(
                    self.write_dataframe,
                    df=df,
                    path=path,
                    partition_cols=partition_cols
                )
                futures[future] = suffix
            
            for future in as_completed(futures):
                suffix = futures[future]
                try:
                    result = future.result()
                    results[suffix] = result
                except Exception as e:
                    logger.error("Failed to write {suffix}")
                    results[suffix] = {'error': str(e)}
        
        return results
    
    def compute_checksum(self, path: str) -> str:
        """
        Compute checksum for dataset to verify integrity.
        
        Args:
            path: Path within bucket
        
        Returns:
            SHA256 checksum of dataset
        """
        full_path = f"{self.base_path}/{path}"
        
        try:
            dataset = pq.ParquetDataset(full_path, filesystem=self.fs)
            
            # Create hash of file paths and sizes
            hasher = hashlib.sha256()
            
            for file in sorted(dataset.files):
                file_info = self.fs.info(f"{self.bucket_name}/{file}")
                hasher.update(f"{file}:{file_info.get('size', 0)}".encode())
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error("Failed to compute checksum for {full_path}")
            raise