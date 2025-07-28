#!/bin/bash
# Entrypoint script for backfill job

# If no arguments provided, show help
if [ $# -eq 0 ]; then
    python -m src.jobs.backfill_job --help
    exit 0
fi

# Execute the backfill job with all arguments
exec python -m src.jobs.backfill_job "$@"