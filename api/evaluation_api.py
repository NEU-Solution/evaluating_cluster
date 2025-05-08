import os
import sys
import time
import asyncio
import requests
from enum import Enum
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, status
from pydantic import BaseModel, HttpUrl
import logging
import uuid
import json
import concurrent.futures

# Add parent directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)


from src.evaluate import evaluate
from src.exp_logging import create_logger, BaseLogger
from const import ConcurrencyStrategy, TrackingBackend, EvaluationRequest, EvaluationResponse, EvaluationStatus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="Model Evaluation API")

# Global thread pool for running CPU-intensive tasks
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Global state for tracking jobs
evaluation_jobs = {}
is_evaluation_running = False
evaluation_queue = []

# Configuration (can be loaded from environment variables)
CONCURRENCY_STRATEGY = ConcurrencyStrategy.QUEUE
DEFAULT_TRACKING_BACKEND = os.getenv("TRACKING_BACKEND", "wandb")

@app.post("/evaluate", response_model=EvaluationResponse)
async def start_evaluation(
    request: EvaluationRequest, 
    background_tasks: BackgroundTasks,
    strategy: ConcurrencyStrategy = Query(
        default=ConcurrencyStrategy.QUEUE,
        description="Strategy for handling concurrent evaluation requests"
    )
):
    """
    Start an evaluation job for the specified model.
    """
    global is_evaluation_running

    job_id = str(uuid.uuid4())
    
    # Check if there's already an evaluation running
    if is_evaluation_running:
        if strategy == ConcurrencyStrategy.REJECT:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An evaluation is already running. Please try again later."
            )
        elif strategy == ConcurrencyStrategy.QUEUE:
            # Queue the job
            evaluation_queue.append((job_id, request))
            logger.info(f"Job {job_id} added to queue. Current queue length: {len(evaluation_queue)}")
            
            # Update job status
            evaluation_jobs[job_id] = {
                "status": "queued", 
                "request": request.dict(),
                "queued_time": time.time()
            }
            
            return EvaluationResponse(
                job_id=job_id,
                status="queued",
                message="Your evaluation has been queued and will start when resources are available"
            )
    
    # Start the evaluation
    is_evaluation_running = True
    evaluation_jobs[job_id] = {
        "status": "running", 
        "request": request.dict(),
        "start_time": time.time()
    }
    
    # Start the evaluation in the background without blocking
    background_tasks.add_task(
        run_evaluation_job, 
        job_id=job_id,
        request=request
    )
    
    # Return immediately with the job ID
    return EvaluationResponse(
        job_id=job_id,
        status="started",
        message="Evaluation job started"
    )

async def run_evaluation_job(job_id: str, request: EvaluationRequest):
    """
    Run the evaluation job and update the status.
    """
    global is_evaluation_running
    
    tracking_backend = request.tracking_backend.value
    logger_instance = None
    tracking_url = None
    
    try:
        logger.info(f"Starting evaluation job {job_id} with tracking backend: {tracking_backend}")
        
        # Initialize the tracking logger
        logger_instance = create_logger(tracking_backend)
        logger_instance.login()
        current_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Initialize tracking run
        run = logger_instance.init_run(
            project=os.getenv("WANDB_PROJECT") if tracking_backend == 'wandb' else os.getenv("MLFLOW_EXPERIMENT_NAME", "model-evaluation"),
            entity=os.getenv("WANDB_ENTITY") if tracking_backend == 'wandb' else None,
            job_type="api_evaluation",
            name=f"api_eval_{current_date}_{job_id}",
            config=request.dict(),
            train_id= request.train_id
        )
        
        # Get tracking URL if available
        if hasattr(logger_instance, 'get_tracking_url'):
            tracking_url = logger_instance.get_tracking_url()
        
        # Update job with tracking URL
        evaluation_jobs[job_id]["tracking_url"] = tracking_url
        
        # Run the evaluation in a separate thread to not block the event loop
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            thread_pool,
            evaluate,
            request.base_model_name,
            request.lora_model_name,
            request.data_version,
            logger_instance,
            request.lora_version,
            request.multi_thread,
            request.llm_backend,
            request.max_workers,
            request.port,
            tracking_backend,
            request.train_id,
            request.num_rounds
        )
        
        # Extract results and metrics (if available)
        run_results = {
            "completed": True,
            "tracking_backend": tracking_backend,
            "tracking_url": tracking_url
        }
        
        # Update job status
        evaluation_jobs[job_id].update({
            "status": "completed",
            "results": run_results,
            "end_time": time.time()
        })
        
        logger.info(f"Evaluation job {job_id} completed successfully")
        
        # Send webhook notification if URL was provided
        if request.webhook_url:
            await send_webhook_notification(
                request.webhook_url,
                job_id=job_id,
                status="completed",
                results=run_results
            )
    
    except Exception as e:
        logger.exception(f"Error in evaluation job {job_id}: {str(e)}")
        # Update job status with error
        evaluation_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": time.time()
        })
        
        # Send webhook notification about failure
        if request.webhook_url:
            await send_webhook_notification(
                request.webhook_url,
                job_id=job_id,
                status="failed",
                error=str(e)
            )
    
    finally:
        # Finish the tracking run
        if logger_instance:
            logger_instance.finish_run()
            
        # Check if there are queued jobs
        is_evaluation_running = False
        await process_queue()

async def process_queue():
    """
    Process the next job in the queue if available.
    """
    global is_evaluation_running, evaluation_queue
    
    if evaluation_queue and not is_evaluation_running:
        job_id, request = evaluation_queue.pop(0)
        is_evaluation_running = True
        
        # Update job status
        evaluation_jobs[job_id].update({
            "status": "running",
            "start_time": time.time()
        })
        
        logger.info(f"Starting queued evaluation job {job_id}")
        
        # Run the evaluation in a new task
        asyncio.create_task(run_evaluation_job(job_id, request))

async def send_webhook_notification(webhook_url: HttpUrl, **data):
    """
    Send a webhook notification with job status and results.
    """
    try:
        payload = {
            "timestamp": time.time(),
            **data
        }
        
        # Run in an executor to avoid blocking
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(
            str(webhook_url),
            json=payload,
            headers={"Content-Type": "application/json"}
        ))
        
        if response.status_code >= 400:
            logger.error(f"Webhook notification failed with status {response.status_code}: {response.text}")
        else:
            logger.info(f"Webhook notification sent successfully to {webhook_url}")
            
    except Exception as e:
        logger.error(f"Error sending webhook notification: {str(e)}")

@app.get("/evaluate/{job_id}", response_model=EvaluationStatus)
async def get_evaluation_status(job_id: str):
    """
    Get the status of an evaluation job.
    """
    if job_id not in evaluation_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation job {job_id} not found"
        )
    
    job = evaluation_jobs[job_id]
    
    return EvaluationStatus(
        job_id=job_id,
        status=job["status"],
        results=job.get("results"),
        error=job.get("error"),
        tracking_url=job.get("tracking_url")
    )

@app.get("/evaluate", response_model=List[EvaluationStatus])
async def get_all_evaluations():
    """
    Get the status of all evaluation jobs.
    """
    return [
        EvaluationStatus(
            job_id=job_id,
            status=job["status"],
            results=job.get("results"),
            error=job.get("error"),
            tracking_url=job.get("tracking_url")
        )
        for job_id, job in evaluation_jobs.items()
    ]

@app.get("/queue")
async def get_queue_status():
    """
    Get the status of the evaluation queue.
    """
    return {
        "queue_length": len(evaluation_queue),
        "is_processing": is_evaluation_running,
        "queued_jobs": [job_id for job_id, _ in evaluation_queue]
    }

@app.get("/")
async def root():
    return {
        "status": "online", 
        "message": "Model Evaluation API is running",
        "supported_tracking": ["wandb", "mlflow"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=23477)