#!/usr/bin/env python3
"""
Test script for the backfill service endpoints.
Can be used locally or against deployed service.
"""

import requests
import json
import sys
from datetime import datetime, timedelta

def test_backfill_service(base_url="http://localhost:8080"):
    """Test the backfill service endpoints."""
    
    print(f"Testing backfill service at: {base_url}")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        print("   ✅ Health check passed")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test general backfill endpoint
    print("\n2. Testing general backfill endpoint...")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3)
    
    payload = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "data_types": ["ohlcv"],
        "batch_size": 5
    }
    
    try:
        response = requests.post(f"{base_url}/backfill", json=payload)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        print("   ✅ General backfill endpoint passed")
    except Exception as e:
        print(f"   ❌ General backfill failed: {e}")
        return False
    
    # Test daily backfill endpoint
    print("\n3. Testing daily backfill endpoint...")
    try:
        response = requests.post(f"{base_url}/backfill/daily")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        print("   ✅ Daily backfill endpoint passed")
    except Exception as e:
        print(f"   ❌ Daily backfill failed: {e}")
        return False
    
    # Test weekly backfill endpoint
    print("\n4. Testing weekly backfill endpoint...")
    try:
        response = requests.post(f"{base_url}/backfill/weekly")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        print("   ✅ Weekly backfill endpoint passed")
    except Exception as e:
        print(f"   ❌ Weekly backfill failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    # Run tests
    success = test_backfill_service(base_url)
    sys.exit(0 if success else 1)