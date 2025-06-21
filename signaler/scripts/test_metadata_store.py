#!/usr/bin/env python3
"""Test script to verify Vertex AI Metadata Store setup."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from google.cloud import aiplatform
from config.settings import VERTEX_AI_CONFIG
from datetime import datetime
import numpy as np

def test_metadata_store():
    """Test Vertex AI Metadata Store functionality."""
    print("🔍 Testing Vertex AI Metadata Store...")

    try:
        # Initialize Vertex AI
        aiplatform.init(
            project=VERTEX_AI_CONFIG['project'],
            location=VERTEX_AI_CONFIG['location'],
            experiment=VERTEX_AI_CONFIG['experiment'],
            experiment_description=VERTEX_AI_CONFIG['experiment_description']
        )
        print("✅ Vertex AI initialized successfully")

        # Create a test run
        run_id = f"test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        with aiplatform.start_run(run=run_id) as run:
            print(f"✅ Created run: {run_id}")

            # Log parameters
            run.log_params({
                "test_param_string": "hello_metadata_store",
                "test_param_int": 42,
                "test_param_float": 3.14,
                "model_type": "test_model",
                "learning_rate": 0.001
            })
            print("✅ Logged parameters")

            # Log metrics
            run.log_metrics({
                "test_metric": 99.9,
                "accuracy": 0.95,
                "loss": 0.05
            })
            print("✅ Logged metrics")

            # Log time series (for Tensorboard)
            for i in range(10):
                run.log_time_series_metrics({
                    "loss/train": 1.0 / (i + 1),
                    "loss/val": 1.2 / (i + 1),
                    "accuracy": i / 10.0
                })
            print("✅ Logged time series metrics")

            print(f"✅ Run resource name: {run.resource_name}")

        # List runs in the experiment
        print("\n📊 Listing experiment runs:")
        runs = aiplatform.ExperimentRun.list(
            experiment=VERTEX_AI_CONFIG['experiment'],
            limit=5
        )

        for i, run in enumerate(runs):
            print(f"  {i+1}. {run.name}")
            if hasattr(run, 'get_metrics'):
                metrics = run.get_metrics()
                print(f"     Metrics: {metrics}")

        print(f"\n✅ Found {len(runs)} runs in experiment")

        # Test Tensorboard
        print("\n🔍 Checking Tensorboard...")
        try:
            tb_name = VERTEX_AI_CONFIG.get('tensorboard', '').split('/')[-1]
            if tb_name:
                print(f"✅ Tensorboard configured: {tb_name}")
                print(f"   View at: https://console.cloud.google.com/vertex-ai/experiments/tensorboard/{tb_name}")
        except:
            print("⚠️  Tensorboard not configured")

        print("\n✅ All metadata store tests passed!")
        return True

    except Exception as e:
        print(f"❌ Metadata store test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Vertex AI API is enabled")
        print("2. Verify service account has 'roles/aiplatform.user' role")
        print("3. Ensure metadata store was created in Terraform")
        return False

def test_example_training_metrics():
    """Simulate logging metrics from a training run."""
    print("\n🎯 Simulating training run metrics...")

    try:
        aiplatform.init(
            project=VERTEX_AI_CONFIG['project'],
            location=VERTEX_AI_CONFIG['location'],
            experiment=VERTEX_AI_CONFIG['experiment']
        )

        run_id = f"simulated-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        with aiplatform.start_run(run=run_id) as run:
            # Log hyperparameters
            run.log_params({
                "model_type": "temporal_gnn",
                "learning_rate": 0.001,
                "batch_size": 32,
                "hidden_dim": 128,
                "num_epochs": 100,
                "num_gnn_layers": 3,
                "dropout_rate": 0.2
            })

            # Simulate training loop
            for epoch in range(10):
                # Simulate metrics
                train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.01)
                val_loss = 1.2 / (epoch + 1) + np.random.normal(0, 0.02)

                run.log_metrics({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "sharpe_ratio": 0.5 + epoch * 0.05,
                    "direction_accuracy": 0.5 + epoch * 0.02
                })

                # Log to time series
                run.log_time_series_metrics({
                    "loss/train": train_loss,
                    "loss/validation": val_loss,
                    "metrics/sharpe": 0.5 + epoch * 0.05
                })

            print(f"✅ Simulated training logged to run: {run_id}")

        return True

    except Exception as e:
        print(f"❌ Failed to simulate training: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Vertex AI Metadata Store Setup\n")

    # Test basic functionality
    if not test_metadata_store():
        sys.exit(1)

    # Test training simulation
    test_example_training_metrics()

    print("\n✅ All tests completed!")
    print("\n📌 Next steps:")
    print("1. View experiments in Cloud Console:")
    print(f"   https://console.cloud.google.com/vertex-ai/experiments")
    print("2. Train a real model with metadata tracking")
    print("3. Compare experiment runs to find the best model")

if __name__ == "__main__":
    main()