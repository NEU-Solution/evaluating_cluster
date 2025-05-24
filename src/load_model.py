import wandb
import mlflow

from huggingface_hub import snapshot_download
# from llm.llm.hugging_face import HuggingFaceLLM
import torch
import gc

from pathlib import Path
import subprocess
import signal
import time
import threading

import logging

import sys 
sys.path.append("..")
from llm import vLLM, OpenAIWrapper
from src.exp_logging import BaseLogger, create_logger

import os 
current_dir = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))



def download_model_regristry(model_name: str, version: str = 'lastest', download_dir: str = 'models', logger: BaseLogger = None, hf_repo: str = None) -> str:
    """
    Download a model from the WandB model registry.
    """

    assert model_name, "Model name can not be empty"
    assert logger, "No logger instance provided"

    # if 'wandb-registry-model' not in model_name:
    #     model_name = 'wandb-registry-model/' + model_name

    # Initialize a W&B run
    
    # Download the model

    download_dir = os.path.join(current_dir, '..', download_dir)
    os.makedirs(download_dir, exist_ok=True)

    if hf_repo is not None:
        # Download from Hugging Face Hub
        artifact_dir = snapshot_download(
            repo_id=hf_repo,
            revision=version,
            cache_dir=download_dir
        )
        logging.info(f"Downloaded model from Hugging Face Hub to {artifact_dir}")
        return artifact_dir

    if logger.tracking_backend == 'wandb':

        # Handle W&B uri download
        artifact_uri = ""
        if 'artifact' in model_name:
            # Handle W&B artifact download
            artifact_uri = model_name
        else:
            if 'wandb-registry' in model_name:
                # Handle W&B model registry download
                artifact_uri = artifact_uri
            else:
                # Handle W&B model download
                artifact_uri = f"wandb-registry-model/{model_name}"
            if version is not None:
                artifact_uri = f"{artifact_uri}:{version}"
            else:
                artifact_uri = f"{artifact_uri}:latest"
            
        # Download the model using wandb API
        artifact = wandb.use_artifact(artifact_uri)
        artifact_dir = artifact.download(root=download_dir)

    elif logger.tracking_backend == 'mlflow':
        # Handle MLflow model download
        if version is None:
            version = "latest"
            
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        # Download via MLflow
        artifact_dir = os.path.join(download_dir, model_name.replace("/", "_"))
        
        if 'models:/' in model_name:
            artifact_uri = model_name
        else:
            artifact_uri = f"models:/{model_name}/{version}" if version else f"models:/{model_name}/latest"

        print(f"Downloading model from MLflow: {artifact_uri}")
        
        
        # Download the model using mlflow
        mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=artifact_dir
        )


        # Extract model name and version from the artifact URI
        if 'models:/' in artifact_uri:
            model_parts = artifact_uri.replace('models:/', '').split('/')
            if len(model_parts) >= 2:
                model_name = '/'.join(model_parts[:-1])
                version = model_parts[-1]
                logging.info(f"Extracted model name: {model_name}, version: {version} from Registry URI")
                logger.set_model_version(model_name, version)
            
            else:
                logging.info(f"Could not parse model name and version from URI: {artifact_uri}")



    else:
        raise ValueError(f"Unsupported logger")
        
    logging.info(f"Downloaded model from {artifact_uri} to {artifact_dir}")
    
    return artifact_dir


def test_connection(port = 8000) -> bool:
    
    try:
        host = f"http://localhost:{port}/v1"
        model_name = "evaluate"
        api_key = 'ngu'

        llm = OpenAIWrapper(model_name=model_name, host=host, api_key=api_key)

        messages = [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]

        response = llm(messages)
        if not response or response == "":
            logging.error("No response from the server.")
            return False
        return True
    except Exception as e:
        logging.error(f"Error connecting to the server: {e}")
        return False


def start_inference_server(base_model: str, lora_path: str, port=8000, max_vram: float = 12):
    """Start the model inference server"""

    max_model_len = 8192
    max_vram = 18

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logging.info(f"Using device: {device}")

    lora_path = os.path.join(current_dir, lora_path)

    logging.info(f"Download base model from {base_model}")
    # Download the model using the Hugging Face Hub snapshot_download function
    model_path = snapshot_download(
        repo_id=base_model
    )

    logging.info(f"Downloaded base model to {model_path}")
    logging.info(f"Starting inference server with model at {lora_path} on port {port}")
    
    # Get the maximum available GPU VRAM
    if device == "cuda":
        try:
            # Get total GPU memory
            total_vram = torch.cuda.get_device_properties(0).total_memory
            # Get current allocated memory
            allocated_vram = torch.cuda.memory_allocated(0)
            # Get cached memory that can be freed
            cached_vram = torch.cuda.memory_reserved(0)
            # Calculate free memory
            free_vram = total_vram - allocated_vram - cached_vram
            
            logging.info(f"Total GPU VRAM: {total_vram / 1024**3:.2f} GB")
            logging.info(f"Free GPU VRAM: {free_vram / 1024**3:.2f} GB")

            total_vram_gb = total_vram / (1024**3)

            low_vram_config = ""
            if total_vram_gb < 13:
                low_vram_config = "--kv-cache-dtype fp8 --enforce-eager"
                logging.info("Using low VRAM configuration")
            
            # Adjust GPU memory utilization based on available memory
            gpu_mem_utilization = min(0.8, (free_vram / total_vram) * 0.8)
            
            # Ensure the utilization does not exceed the max_vram limit
            
            # Ratio
            if max_vram < 1:
                gpu_mem_utilization = min(gpu_mem_utilization, max_vram)
            else:
                # Actual memory
                gpu_mem_utilization = min(gpu_mem_utilization, max_vram / (total_vram / 1024**3))

            logging.info(f"Setting GPU memory utilization to: {gpu_mem_utilization:.2f}")
        except Exception as e:
            logging.warning(f"Failed to get GPU memory info: {e}")
            gpu_mem_utilization = 0.8
    else:
        logging.warning("CUDA not available, running on CPU")
        gpu_mem_utilization = 0.0

    gpu_mem_utilization = min(gpu_mem_utilization, 0.8)

    # Example command to start an inference server (adjust based on your actual server command)
    server_command = f"vllm serve {base_model} --lora-modules evaluate={lora_path} --max_model-len {max_model_len} --gpu-memory-utilization {gpu_mem_utilization} --enable-lora {low_vram_config}  --max-lora-rank 64 --served-model-name evaluate --port {port}"
    logging.info(server_command)

    # Start the server as a subprocess
    server_process = subprocess.Popen(
        server_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # This allows us to terminate the process group later
        bufsize=1,  # Line buffered
        universal_newlines=True,  # Text mode
        env=os.environ.copy()
    )

    def log_output(pipe, prefix):
        for line in iter(pipe.readline, ''):
            logging.info(f"{prefix}: {line.strip()}")

    stdout_thread = threading.Thread(target=log_output, args=(server_process.stdout, "SERVER-OUT"))
    stderr_thread = threading.Thread(target=log_output, args=(server_process.stderr, "SERVER-ERR"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for server to start, at maximum of 2 minutes
    time.sleep(40)  # Adjust as needed

    max_tries = 12
    while max_tries > 0:
        if test_connection(port):
            logging.info("Server started successfully")
            break
        else:
            logging.info("Server not ready yet, retrying...")
            time.sleep(10)
            max_tries -= 1
    
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


