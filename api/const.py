from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, HttpUrl

class ConcurrencyStrategy(str, Enum):
    REJECT = "reject"
    QUEUE = "queue"

class TrackingBackend(str, Enum):
    WANDB = "wandb"
    MLFLOW = "mlflow"

# Request model
class EvaluationRequest(BaseModel):
    base_model_name: str
    lora_model_name: str
    data_version: str = "latest"
    lora_version: Optional[str] = "latest"
    multi_thread: bool = True
    llm_backend: str = 'vllm'
    max_workers: int = 2
    port: int = 8000
    tracking_backend: TrackingBackend = TrackingBackend.WANDB
    train_id: Optional[str] = None
    webhook_url: Optional[HttpUrl] = None
    num_rounds: int = 3

# Response models
class EvaluationResponse(BaseModel):
    job_id: str
    status: str
    message: str

class EvaluationStatus(BaseModel):
    job_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tracking_url: Optional[str] = None