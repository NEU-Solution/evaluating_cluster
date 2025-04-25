from src.evaluate import evaluate
import datetime

from types import SimpleNamespace

import os 
import wandb
from dotenv import load_dotenv
from src.exp_logging import BaseLogger, create_logger


load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

wandb.login(key = WANDB_API_KEY)

config = SimpleNamespace(
    # Model configuration
    base_model='Qwen/Qwen2.5-1.5B-Instruct',
    lora_model='wandb-registry-model/initial-sft',
    data_version='latest',
    llm_bankend='vllm',  # 'vllm' or 'huggingface' (exp)
    alias='v0',
    
    # Tracking configuration
    tracking_backend='wandb',  # 'wandb' or 'mlflow'
    
    # WandB specific config
    wandb_project=os.getenv("WANDB_PROJECT"),
    wandb_entity=os.getenv("WANDB_ENTITY"),
    
    # MLflow specific config
    mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "model-evaluation"),
)

if __name__ == "__main__":
    # Run the evaluation
    logger = create_logger(config.tracking_backend)
    logger.login()
    
    # Initialize the tracking run
    run_name = f"eval_vi_llm_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    if config.tracking_backend == 'wandb':
        run = logger.init_run(
            project=config.wandb_project,
            entity=config.wandb_entity,
            job_type="evaluate",
            config=vars(config),
            name=run_name
        )
    else:  # mlflow
        run = logger.init_run(
            project=config.mlflow_experiment_name,
            job_type="evaluate",
            config=vars(config),
            name=run_name
        )
    
    # Run the evaluation
    try:
        evaluate(
            base_model_name=config.base_model,
            lora_name=config.lora_model,
            data_version=config.data_version,
            model_version=config.alias,
            llm_bankend=config.llm_bankend,
            logger=logger,
            tracking_backend=config.tracking_backend,
        )
    finally:
        logger.finish_run()
        
    print("Evaluation completed.")