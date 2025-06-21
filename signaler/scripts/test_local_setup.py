#!/usr/bin/env python3
"""Test script to verify GCP setup for local development."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud import secretmanager

# Load environment variables
load_dotenv()

def test_bigquery():
    """Test BigQuery access."""
    print("🔍 Testing BigQuery access...")
    try:
        client = bigquery.Client()

        # List datasets
        datasets = list(client.list_datasets())
        print(f"✅ Found {len(datasets)} datasets")

        # Check for trading_signals dataset
        dataset_id = "trading_signals"
        dataset_ref = client.dataset(dataset_id)

        try:
            dataset = client.get_dataset(dataset_ref)
            print(f"✅ Dataset '{dataset_id}' exists")

            # List tables
            tables = list(client.list_tables(dataset_ref))
            print(f"✅ Found {len(tables)} tables:")

            # Expected tables
            expected_tables = [
                'raw_ohlcv',
                'technical_indicators',
                'macro_indicators',
                'sentiment_data',
                'temporal_features',
                'stock_metadata',
                'predictions',
                'model_metadata',
                'job_logs'
            ]

            for table in tables:
                status = "✓" if table.table_id in expected_tables else "?"
                print(f"   {status} {table.table_id}")

            # Check for missing tables
            existing_tables = {table.table_id for table in tables}
            missing_tables = set(expected_tables) - existing_tables
            if missing_tables:
                print(f"⚠️  Missing tables: {', '.join(missing_tables)}")

        except Exception as e:
            print(f"❌ Dataset '{dataset_id}' not found: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ BigQuery test failed: {e}")
        print("   Make sure GOOGLE_APPLICATION_CREDENTIALS is set correctly")
        return False

def test_secret_manager():
    """Test Secret Manager access."""
    print("\n🔍 Testing Secret Manager access...")
    try:
        client = secretmanager.SecretManagerServiceClient()

        # Try to access the Alpha Vantage key
        project_id = os.getenv("GCP_PROJECT_ID", "trading-signals-420-69")
        secret_name = f"projects/{project_id}/secrets/alpha-vantage-api-key/versions/latest"

        try:
            response = client.access_secret_version(request={"name": secret_name})
            secret_value = response.payload.data.decode("UTF-8")
            if secret_value == "YOUR_ALPHA_VANTAGE_API_KEY":
                print(f"⚠️  Alpha Vantage API key is still the default value")
                print("   Update it with: gcloud secrets versions add alpha-vantage-api-key --data-file=<(echo 'YOUR_REAL_KEY')")
            else:
                print(f"✅ Alpha Vantage API key retrieved (length: {len(secret_value)})")
            return True
        except Exception as e:
            print(f"⚠️  Could not access secret: {e}")
            print("   This is OK if you haven't created it yet")
            return True  # Not critical for initial setup

    except Exception as e:
        print(f"❌ Secret Manager test failed: {e}")
        return False

def test_credentials():
    """Test that credentials are properly set."""
    print("🔍 Testing credentials...")

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
        print("   Set it with: export GOOGLE_APPLICATION_CREDENTIALS=~/trading-system-key.json")
        return False

    # Expand ~ to home directory
    creds_path = os.path.expanduser(creds_path)

    if not os.path.exists(creds_path):
        print(f"❌ Credentials file not found: {creds_path}")
        print("   Create it with:")
        print("   gcloud iam service-accounts keys create ~/trading-system-key.json \\")
        print("     --iam-account=trading-system@trading-signals-420-69.iam.gserviceaccount.com")
        return False

    print(f"✅ Credentials file found: {creds_path}")

    # Check if it's valid JSON
    try:
        import json
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
            if 'project_id' in creds_data:
                print(f"✅ Valid service account key for project: {creds_data['project_id']}")
    except Exception as e:
        print(f"⚠️  Could not parse credentials file: {e}")

    return True

def test_environment():
    """Test environment variables."""
    print("\n🔍 Testing environment variables...")

    env_vars = {
        'GCP_PROJECT_ID': os.getenv('GCP_PROJECT_ID', 'trading-signals-420-69'),
        'BQ_DATASET': os.getenv('BQ_DATASET', 'trading_signals'),
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY', '')
    }

    all_good = True
    for var, value in env_vars.items():
        if value and value != '':
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set (optional for local testing)")
            if var == 'ALPHA_VANTAGE_API_KEY':
                all_good = False

    return all_good

def main():
    """Run all tests."""
    print("🚀 Testing GCP Setup for Trading Signal System\n")

    all_passed = True

    # Test credentials first
    if not test_credentials():
        all_passed = False
        print("\n❌ Fix credential issues before continuing")
        sys.exit(1)

    # Test environment
    test_environment()  # Don't fail on this, just warn

    # Test BigQuery
    if not test_bigquery():
        all_passed = False

    # Test Secret Manager
    if not test_secret_manager():
        all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed! You're ready to run ingestion locally.")
        print("\nNext steps:")
        print("1. Make sure your .env file is configured")
        print("2. Run a test backfill:")
        print("   python -m src.jobs.backfill_job --start-date 2024-01-01 --end-date 2024-01-07")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Run: ./scripts/terraform_minimal.sh")
        print("2. Create service account key")
        print("3. Set GOOGLE_APPLICATION_CREDENTIALS")
        sys.exit(1)

if __name__ == "__main__":
    main()