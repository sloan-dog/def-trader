#!/usr/bin/env python3
"""
Test script for the ingestion HTTP server.
"""
import requests
import time
import json

def test_ingestion_server():
    """Test the ingestion server endpoints."""
    base_url = "http://localhost:8080"
    
    print("Testing ingestion server...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"Root endpoint: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing root endpoint: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"\nHealth endpoint: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/status")
        print(f"\nStatus endpoint: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing status endpoint: {e}")
    
    # Test run endpoint
    try:
        response = requests.post(f"{base_url}/run")
        print(f"\nRun endpoint: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing run endpoint: {e}")
    
    # Test stop endpoint (should fail if no job running)
    try:
        response = requests.post(f"{base_url}/stop")
        print(f"\nStop endpoint: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing stop endpoint: {e}")
    
    # Wait a bit and check status again
    print("\nWaiting 5 seconds...")
    time.sleep(5)
    
    try:
        response = requests.get(f"{base_url}/status")
        print(f"Updated status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing updated status: {e}")

if __name__ == "__main__":
    test_ingestion_server() 