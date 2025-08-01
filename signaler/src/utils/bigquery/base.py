"""
Base BigQuery client with core functionality.
"""
from typing import Optional
from google.cloud import bigquery
from src.utils import logger
from config.settings import GCP_PROJECT_ID, BQ_DATASET, BQ_LOCATION


class BigQueryBase:
    """Base class for BigQuery operations."""

    def __init__(self, project_id: str = GCP_PROJECT_ID):
        """
        Initialize BigQuery client.

        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.dataset_id = BQ_DATASET
        self.location = BQ_LOCATION
        self.dataset_ref = f"{project_id}.{BQ_DATASET}"

        # Initialize client
        self._client: Optional[bigquery.Client] = None

    @property
    def client(self) -> bigquery.Client:
        """Get or create BigQuery client."""
        if self._client is None:
            self._client = bigquery.Client(project=self.project_id)
            logger.info(f"Initialized BigQuery client for project: {self.project_id}")
        return self._client

    def get_dataset_reference(self) -> bigquery.DatasetReference:
        """Get dataset reference."""
        return self.client.dataset(self.dataset_id)

    def get_table_reference(self, table_name: str) -> bigquery.TableReference:
        """Get table reference."""
        dataset_ref = self.get_dataset_reference()
        return dataset_ref.table(table_name)

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists, False otherwise
        """
        try:
            table_ref = self.get_table_reference(table_name)
            self.client.get_table(table_ref)
            return True
        except Exception:
            return False

    def dataset_exists(self) -> bool:
        """
        Check if the dataset exists.

        Returns:
            True if dataset exists, False otherwise
        """
        try:
            self.client.get_dataset(self.get_dataset_reference())
            return True
        except Exception:
            return False