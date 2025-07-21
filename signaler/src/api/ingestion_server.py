"""
Simple HTTP server wrapper for daily ingestion job.
This allows the ingestion job to run as a Cloud Run service.
"""
import os
import threading
import time
import atexit
from datetime import datetime
from flask import Flask, jsonify, request
from loguru import logger

from src.shared_logging import setup_logging, log_exception
from src.jobs.daily_ingestion import DailyIngestionJob

# Configure logging
setup_logging(level="INFO", app_name="signaler-ingestion-service")

app = Flask(__name__)

# Thread-safe job status tracking
class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._job_status = {
            "last_run": None,
            "is_running": False,
            "last_error": None,
            "last_success": None,
            "current_thread": None,
            "start_time": None,
            "duration": None
        }
    
    def get_status(self):
        """Get a copy of the current job status."""
        with self._lock:
            return self._job_status.copy()
    
    def update_status(self, **kwargs):
        """Update job status in a thread-safe manner."""
        with self._lock:
            self._job_status.update(kwargs)
    
    def is_job_running(self):
        """Check if job is currently running."""
        with self._lock:
            return self._job_status["is_running"]

# Global job manager
job_manager = JobManager()

def run_ingestion_job():
    """Run the daily ingestion job in a separate thread."""
    start_time = datetime.now()
    
    try:
        job_manager.update_status(
            is_running=True,
            last_run=start_time.isoformat(),
            last_error=None,
            start_time=start_time.isoformat(),
            current_thread=threading.current_thread().ident
        )
        
        logger.info("Starting daily ingestion job", 
                   thread_id=threading.current_thread().ident)
        
        job = DailyIngestionJob()
        results = job.run()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if results['overall_success']:
            job_manager.update_status(
                last_success=end_time.isoformat(),
                duration=duration
            )
            logger.info("Daily ingestion completed successfully", 
                       duration=duration,
                       steps_completed=len(results.get('steps', {})),
                       thread_id=threading.current_thread().ident)
        else:
            error_msg = results.get('error', 'Unknown error')
            job_manager.update_status(
                last_error=error_msg,
                duration=duration
            )
            logger.error("Daily ingestion failed", 
                        duration=duration,
                        error=error_msg,
                        steps=results.get('steps', {}),
                        thread_id=threading.current_thread().ident)
            
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        job_manager.update_status(
            last_error=str(e),
            duration=duration
        )
        log_exception("Fatal error in daily ingestion job", 
                     exception=e,
                     thread_id=threading.current_thread().ident)
    finally:
        job_manager.update_status(
            is_running=False,
            current_thread=None
        )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run."""
    status = job_manager.get_status()
    
    # Consider unhealthy if job has been running for more than 2 hours
    is_healthy = True
    if status["is_running"] and status["start_time"]:
        start_time = datetime.fromisoformat(status["start_time"])
        duration = (datetime.now() - start_time).total_seconds()
        if duration > 7200:  # 2 hours
            is_healthy = False
    
    return jsonify({
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "job_status": status
    })


@app.route('/run', methods=['POST'])
def trigger_ingestion():
    """Trigger the daily ingestion job."""
    if job_manager.is_job_running():
        status = job_manager.get_status()
        return jsonify({
            "status": "error",
            "message": "Job is already running",
            "job_status": status
        }), 409
    
    # Start job in background thread
    thread = threading.Thread(target=run_ingestion_job, name="ingestion-job")
    thread.daemon = True
    thread.start()
    
    logger.info("Daily ingestion job triggered", 
               thread_id=thread.ident,
               request_id=request.headers.get('X-Request-ID', 'unknown'))
    
    return jsonify({
        "status": "started",
        "message": "Daily ingestion job started",
        "timestamp": datetime.now().isoformat(),
        "thread_id": thread.ident
    })


@app.route('/status', methods=['GET'])
def get_status():
    """Get current job status."""
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "job_status": job_manager.get_status()
    })


@app.route('/stop', methods=['POST'])
def stop_job():
    """Stop the currently running job (if any)."""
    status = job_manager.get_status()
    
    if not status["is_running"]:
        return jsonify({
            "status": "error",
            "message": "No job is currently running"
        }), 404
    
    # Note: This is a basic implementation. In production, you might want
    # to implement proper job cancellation using threading.Event or similar
    logger.warning("Job stop requested - this is a basic implementation")
    
    return jsonify({
        "status": "stopping",
        "message": "Job stop requested (basic implementation)",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "service": "signaler-ingestion-service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status", 
            "run": "/run (POST)",
            "stop": "/stop (POST)"
        },
        "timestamp": datetime.now().isoformat()
    })


def cleanup_on_exit():
    """Cleanup function to run when the server shuts down."""
    logger.info("Shutting down ingestion service")


if __name__ == '__main__':
    # Register cleanup function
    atexit.register(cleanup_on_exit)
    
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.environ.get('PORT', 8080))
    
    logger.info(f"Starting ingestion service on port {port}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False) 