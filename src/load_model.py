import wandb
import mlflow

from transformers import AutoModelForCausalLM
# from llm.llm.hugging_face import HuggingFaceLLM
import torch
import gc

from pathlib import Path
import subprocess
import signal
import time


import logging

import sys 
sys.path.append("..")
from llm import vLLM
from src.logging import BaseLogger, create_logger

import os 
current_dir = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

wandb.login(key = WANDB_API_KEY)

def download_model_regristry(model_name: str, version: str = None, download_dir: str = 'models', logger: BaseLogger = None) -> str:
    """
    Download a model from the WandB model registry.
    """

    assert model_name, "Model name can not be empty"
    assert logger, "No logger instance provided"

    # if 'wandb-registry-model' not in model_name:
    #     model_name = 'wandb-registry-model/' + model_name

    # Initialize a W&B run
    
    # Download the model
    artifact = wandb.use_artifact(f"{model_name}:{version}" if version else f"{model_name}:latest")
    
    download_dir = os.path.join('../', download_dir)
    os.makedirs(download_dir, exist_ok=True)

    if logger.tracking_backend == 'wandb':
        if 'wandb-registry-model' not in model_name:
            model_name = 'wandb-registry-model/' + model_name
            
        # Download the model using wandb API
        artifact = wandb.use_artifact(
            f"{model_name}:{version}" if version else f"{model_name}:latest"
        )
        artifact_dir = artifact.download(root=download_dir)

    elif logger.tracking_backend == 'mlflow':
        # Handle MLflow model download
        if version is None:
            version = "latest"
            
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        # Download via MLflow
        artifact_dir = os.path.join(download_dir, model_name.replace("/", "_"))
        registered_model = mlflow.register_model(
            f"models:/{model_name}/{version}",
            model_name
        )
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{version}",
            dst_path=artifact_dir
        )
    else:
        raise ValueError(f"Unsupported logger")
        
    logging.info(f"Downloaded model {model_name} version {version} to {artifact_dir}")
    
    return artifact_dir


def start_inference_server(base_model: str, lora_path: str, port=8000):
    """Start the model inference server"""

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logging.info(f"Using device: {device}")

    lora_path = os.path.join(current_dir, lora_path)

    logging.info(f"Download base model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    del model
    gc.collect()
    logging.info(f"Starting inference server with model at {lora_path} on port {port}")
    
    # Example command to start an inference server (adjust based on your actual server command)
    server_command = f"vllm serve {base_model} --lora-modules evaluate={lora_path} --max_model-len 2048 --gpu-memory-utilization 0.5 --enable-lora  --max-lora-rank 64 --served-model-name evaluate --port {port}"
    logging.info(server_command)

    # Start the server as a subprocess
    server_process = subprocess.Popen(
        server_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # This allows us to terminate the process group later
    )
    
    # Wait for server to start
    time.sleep(60)  # Adjust as needed
    
    return server_process

def terminate_server(server_process):
    """Terminate the inference server"""
    logging.info("Terminating the inference server")
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait(timeout=10)
        logging.info("Server terminated successfully")
    except subprocess.TimeoutExpired:
        logging.warning("Server did not terminate gracefully, forcing termination")
        os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
    except Exception as e:
        logging.error(f"Error terminating server: {e}")


# def load_huggingface_model(model_name: str, lora_path: str) -> HuggingFaceLLM:
#     # Check device
#     if torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"

#     logging.info(f"Using device: {device}")

#     lora_path = os.path.join(current_dir, lora_path)

#     llm = HuggingFaceLLM(
#         model_name=model_name,
#         lora_name=lora_path,
#     )
#     return llm




def load_model_from_registry(model_name: str, version: str = None, logger: BaseLogger = None) -> tuple:
    """
    Load a model from the model registry.
    
    Args:
        model_name: Name of the model to load
        version: Model version
        tracking_backend: Which tracking system to use ('wandb' or 'mlflow')
        logger_instance: Logger instance to use
    
    Returns:
        Loaded model
    """
    artifact_dir = download_model_regristry(
        model_name, 
        version=version, 
        logger=logger
    )
    
    # Load the model
    logging.info(f"Loading model from {artifact_dir}")
    return vLLM(model_name=model_name, lora_path=artifact_dir)

if __name__ == '__main__':

    tracking_backend = os.getenv("TRACKING_BACKEND", "wandb")
    logger_instance = create_logger(tracking_backend)
    logger_instance.login()
    
    try:
        # Initialize tracking run
        run = logger_instance.init_run(
            project=os.getenv("WANDB_PROJECT") if tracking_backend == "wandb" else os.getenv("MLFLOW_EXPERIMENT_NAME", "model-registry"),
            entity=os.getenv("WANDB_ENTITY") if tracking_backend == "wandb" else None,
            job_type="model_download"
        )
        
        # Download the model
        model_path = download_model_regristry(
            'first-collection', 
            logger=logger_instance
        )
        
        logging.info(f"Model downloaded to {model_path}")
        
    except Exception as e:
        logging.error(f"Error in model download: {str(e)}")
    finally:
        logger_instance.finish_run()


