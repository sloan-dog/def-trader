# Shared Logging Module

A simple shared logging module for all Cloud Run apps in the monorepo. Provides consistent, structured JSON logging that works well with Google Cloud Logging.

## Features

- **Structured JSON logging**: Each log entry is a single JSON object that Google Cloud treats as one log entry
- **Exception handling**: Full tracebacks are captured in structured format
- **Context-aware**: Automatically detects Cloud Run environment and configures logging appropriately
- **App identification**: Includes app name in logs for better organization
- **Fallback support**: Graceful fallback to simple format if JSON creation fails

## Quick Start

### 1. Import the shared module

In any Python file that needs logging:

```python
from src.shared_logging import setup_logging, log_exception
from loguru import logger
```

### 2. Configure logging

```python
# Configure logging (auto-detects Cloud Run environment)
setup_logging(
    level="INFO",
    app_name="my-app"  # Optional, will be auto-detected in Cloud Run
)
```

### 3. Use structured logging

```python
# Basic logging
logger.info("Application started")
logger.warning("Something to watch out for")
logger.error("Something went wrong")

# Structured logging with context
logger.info("Processing user data", 
           user_id=123,
           records_processed=150,
           success_rate=0.95)

# Exception logging
try:
    raise ValueError("Something bad happened")
except Exception as e:
    log_exception("Failed to process data", exception=e, user_id=123)
```

## Automatic Migration

Use the provided script to automatically migrate existing apps:

```bash
# From the signaler directory
python apply_shared_logging.py ../another-app
```

This will:
1. Find all Python files in the target app
2. Update imports to use shared logging
3. Create a test script to verify the migration

## Output Format

### JSON Format (Cloud Run)
```json
{
  "timestamp": "2025-07-20T17:25:39.042179-07:00",
  "severity": "INFO",
  "message": "Processing ticker data",
  "sourceLocation": {
    "file": "my_app.py",
    "line": 42,
    "function": "process_data"
  },
  "extra": {
    "ticker": "AAPL",
    "records_processed": 150,
    "success_rate": 0.95,
    "app_name": "my-app"
  },
  "module": "my_app"
}
```

### Human Format (Local Development)
```
2025-07-20 17:25:39.042 | INFO     | my-app | my_app:process_data:42 - Processing ticker data
```

## Configuration Options

### setup_logging()

- `level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `enable_json`: Force JSON format (None = auto-detect)
- `log_file`: Optional log file path
- `rotation`: Log file rotation policy
- `retention`: Log file retention policy
- `app_name`: Application name for logging context

### Environment Variables

- `K_SERVICE`: Cloud Run service name (auto-detected)
- `GOOGLE_CLOUD_PROJECT`: GCP project ID (auto-detected)
- `APP_NAME`: Custom app name (fallback)

## Benefits for Google Cloud Console

- **No more fragmented logs**: Each log entry is one complete JSON object
- **Better searchability**: Structured fields make logs easier to filter and search
- **Cleaner error display**: Exceptions are properly structured instead of scattered
- **More context**: Additional fields provide better debugging information

## Testing

Each app gets a test script to verify logging works:

```bash
cd your-app
python test_shared_logging.py
```

## Migration Checklist

For each Cloud Run app:

1. ✅ Run the migration script: `python apply_shared_logging.py ../app-name`
2. ✅ Test the logging: `cd ../app-name && python test_shared_logging.py`
3. ✅ Update Dockerfile to include shared_logging.py
4. ✅ Deploy and verify logs in Google Cloud Console

## Dockerfile Integration

Make sure your Dockerfile copies the shared logging module:

```dockerfile
# Copy shared logging module
COPY src/shared_logging.py /app/src/shared_logging.py
```

Or add it to your build context and copy it appropriately. 