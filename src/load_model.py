import wandb
from transformers import AutoModelForCausalLM
# from llm.llm.hugging_face import HuggingFaceLLM
import torch
import gc

from pathlib import Path
import subprocess
import signal
import time


import logging
logger = logging.getLogger(__name__)

import sys 
sys.path.append("..")
from llm import vLLM

import os 
current_dir = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

wandb.login(key = WANDB_API_KEY)

def download_model_regristry(model_name: str, version: str = None, download_dir: str = 'models') -> str:
    """
    Download a model from the WandB model registry.
    """
    if 'wandb-registry-model' not in model_name:
        model_name = 'wandb-registry-model/' + model_name

    # Initialize a W&B run
    
    # Download the model
    artifact = wandb.use_artifact(f"{model_name}:{version}" if version else f"{model_name}:latest")
    
    download_dir = os.path.join('../', download_dir)
    os.makedirs(download_dir, exist_ok=True)
    
    artifact_dir = artifact.download(root=download_dir)
    logger.info(f"Model downloaded to {artifact_dir}")    
    # Finish the W&B run
    # run.finish()
    
    return artifact_dir


def start_inference_server(base_model: str, lora_path: str, port=8000):
    """Start the model inference server"""

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    lora_path = os.path.join(current_dir, lora_path)

    logger.info(f"Download base model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    del model
    gc.collect()
    logger.info(f"Starting inference server with model at {lora_path} on port {port}")
    
    # Example command to start an inference server (adjust based on your actual server command)
    server_command = f"vllm serve {base_model} --lora-modules evaluate={lora_path} --max_model-len 2048 --gpu-memory-utilization 0.5 --enable-lora  --max-lora-rank 64 --served-model-name evaluate --port {port}"
    logger.info(server_command)

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
    logger.info("Terminating the inference server")
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait(timeout=10)
        logger.info("Server terminated successfully")
    except subprocess.TimeoutExpired:
        logger.warning("Server did not terminate gracefully, forcing termination")
        os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
    except Exception as e:
        logger.error(f"Error terminating server: {e}")


# def load_huggingface_model(model_name: str, lora_path: str) -> HuggingFaceLLM:
#     # Check device
#     if torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"

#     logger.info(f"Using device: {device}")

#     lora_path = os.path.join(current_dir, lora_path)

#     llm = HuggingFaceLLM(
#         model_name=model_name,
#         lora_name=lora_path,
#     )
#     return llm




def load_model_from_registry(model_name: str, version: str = None) -> tuple:
    """
    Load a model from the WandB model registry.
    """
    artifact_dir = download_model_regristry(model_name, version)
    
    # Load the model using the appropriate library (e.g., Hugging Face Transformers)
    # This is a placeholder; replace with actual model loading code
    logging.info("Loaded model from " + artifact_dir)

    # Tinh sau
    return vLLM(model_name=model_name, lora_path=artifact_dir)

if __name__ == '__main__':

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="download")

    download_model_regristry(f'wandb-registry-model/first-collection')

    wandb.finish()


