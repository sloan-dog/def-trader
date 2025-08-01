"""
Administrative BigQuery operations for dataset and table management.
"""
from typing import Dict, List, Optional
from google.cloud import bigquery
from src.utils import logger
from .base import BigQueryBase
from .schemas import (
    get_table_schemas,
    get_partitioning_config,
    get_clustering_fields
)


class BigQueryAdmin(BigQueryBase):
    """Administrative operations for BigQuery."""

    def create_dataset_if_not_exists(self) -> None:
        """Create BigQuery dataset if it doesn't exist."""
        dataset_id = f"{self.project_id}.{self.dataset_id}"

        try:
            self.client.get_dataset(dataset_id)
            logger.info(f"Dataset {dataset_id} already exists")
        except Exception:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = self.location
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset_id}")

    def create_tables(self) -> None:
        """Create all required tables with schemas."""
        schemas = get_table_schemas()

        for table_name, schema in schemas.items():
            self.create_table(table_name, schema)

    def create_table(
            self,
            table_name: str,
            schema: List[bigquery.SchemaField],
            clustering_fields: Optional[List[str]] = None,
            partition_field: Optional[str] = None
    ) -> None:
        """
        Create a single table with schema.

        Args:
            table_name: Name of the table
            schema: Table schema
            clustering_fields: Fields to cluster by
            partition_field: Field to partition by
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"

        try:
            self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except Exception:
            table = bigquery.Table(table_id, schema=schema)

            # Add partitioning
            partition_config = get_partitioning_config(table_name)
            if partition_config or partition_field:
                field = partition_field or partition_config.get('field')
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field=field
                )

            # Add clustering
            cluster_fields = clustering_fields or get_clustering_fields(table_name)
            if cluster_fields:
                table.clustering_fields = cluster_fields

            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")

    def drop_table(self, table_name: str, not_found_ok: bool = True) -> None:
        """
        Drop a table.

        Args:
            table_name: Name of the table to drop
            not_found_ok: If True, don't raise error if table doesn't exist
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        self.client.delete_table(table_id, not_found_ok=not_found_ok)
        logger.info(f"Dropped table {table_id}")

    def copy_table(
            self,
            source_table: str,
            destination_table: str,
            write_disposition: str = 'WRITE_TRUNCATE'
    ) -> None:
        """
        Copy a table.

        Args:
            source_table: Source table name
            destination_table: Destination table name
            write_disposition: How to write to destination
        """
        source_table_id = f"{self.project_id}.{self.dataset_id}.{source_table}"
        dest_table_id = f"{self.project_id}.{self.dataset_id}.{destination_table}"

        job_config = bigquery.CopyJobConfig(
            write_disposition=write_disposition
        )

        job = self.client.copy_table(
            source_table_id,
            dest_table_id,
            job_config=job_config
        )
        job.result()

        logger.info(f"Copied {source_table} to {destination_table}")

    def get_table_info(self, table_name: str) -> Dict[str, any]:
        """
        Get information about a table.

        Args:
            table_name: Table name

        Returns:
            Dictionary with table information
        """
        table_ref = self.get_table_reference(table_name)
        table = self.client.get_table(table_ref)

        return {
            'table_id': table.table_id,
            'created': table.created,
            'modified': table.modified,
            'num_rows': table.num_rows,
            'num_bytes': table.num_bytes,
            'schema': [field.to_api_repr() for field in table.schema],
            'partitioning': table.time_partitioning._properties if table.time_partitioning else None,
            'clustering': table.clustering_fields,
        }

    def update_table_schema(
            self,
            table_name: str,
            new_fields: List[bigquery.SchemaField]
    ) -> None:
        """
        Add new fields to table schema.

        Args:
            table_name: Table name
            new_fields: New fields to add
        """
        table_ref = self.get_table_reference(table_name)
        table = self.client.get_table(table_ref)

        original_schema = table.schema
        new_schema = original_schema[:]
        new_schema.extend(new_fields)

        table.schema = new_schema
        table = self.client.update_table(table, ["schema"])

        logger.info(f"Updated schema for {table_name}, added {len(new_fields)} fields")

    def create_or_update_view(
            self,
            view_name: str,
            query: str,
            description: Optional[str] = None
    ) -> None:
        """
        Create or update a view.

        Args:
            view_name: Name of the view
            query: SQL query for the view
            description: Optional description
        """
        view_id = f"{self.project_id}.{self.dataset_id}.{view_name}"
        view = bigquery.Table(view_id)
        view.view_query = query

        if description:
            view.description = description

        try:
            # Try to update existing view
            existing_view = self.client.get_table(view_id)
            existing_view.view_query = query
            if description:
                existing_view.description = description
            self.client.update_table(existing_view, ["view_query", "description"])
            logger.info(f"Updated view {view_name}")
        except:
            # Create new view
            self.client.create_table(view)
            logger.info(f"Created view {view_name}")