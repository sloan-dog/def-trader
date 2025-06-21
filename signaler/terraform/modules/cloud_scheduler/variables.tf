# Vertex AI Module

This module manages Vertex AI resources for the Trading Signal System.

## Usage

```hcl
module "vertex_ai" {
source = "./modules/vertex_ai"

project_id      = var.project_id
region          = var.region
service_account = google_service_account.trading_system.email

training_config = {
machine_type      = "n1-standard-8"
accelerator_type  = "NVIDIA_TESLA_T4"
accelerator_count = 1
boot_disk_size_gb = 100
}

create_workbench_instance = false
}
```

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|----------|
| project_id | GCP Project ID | `string` | - | yes |
| region | GCP region for Vertex AI resources | `string` | - | yes |
| service_account | Service account email for Vertex AI | `string` | - | yes |
| training_config | Configuration for Vertex AI training jobs | `object` | See below | no |
| create_workbench_instance | Whether to create a Vertex AI Workbench instance | `bool` | `false` | no |
| labels | Labels to apply to Vertex AI resources | `map(string)` | `{}` | no |

## Outputs

| Name | Description |
|------|-------------|
| staging_bucket | Name of the Vertex AI staging bucket |
| tensorboard_id | ID of the Vertex AI Tensorboard |
| tensorboard_name | Name of the Vertex AI Tensorboard |
| metadata_store_id | ID of the Vertex AI Metadata Store |
| training_job_spec | Training job specification for custom training jobs |
| workbench_instance_name | Name of the Workbench instance (if created) |

## Training Configuration

The `training_config` object supports:

- `machine_type`: GCP machine type (e.g., "n1-standard-8", "n1-highmem-16")
- `accelerator_type`: GPU type (e.g., "NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100")
- `accelerator_count`: Number of GPUs
- `boot_disk_size_gb`: Boot disk size in GB
- `training_image_uri`: Custom training container image (optional)

## Resources Created

1. **Storage Bucket**: For staging training artifacts and model outputs
2. **Tensorboard**: For experiment tracking and visualization
3. **Metadata Store**: For tracking ML experiments and lineage
4. **Workbench Instance** (optional): Jupyter notebook environment for development

## Example Training Job

After applying this module, you can submit training jobs using:

```bash
gcloud ai custom-jobs create \
--region=us-central1 \
--display-name=trading-signal-training \
--config=training-config.yaml
```

Where `training-config.yaml` uses the `training_job_spec` output from this module.