from src.evaluate import evaluate
import datetime

from types import SimpleNamespace

import os 
import wandb
from dotenv import load_dotenv
load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

wandb.login(key = WANDB_API_KEY)

config = SimpleNamespace(

    wandb_project=WANDB_PROJECT,
    wandb_entity=WANDB_ENTITY,
    wandb_model='wandb-registry-model/initial-sft',
    base_model='Qwen/Qwen2.5-1.5B-Instruct',
    data_version='lastest',
    llm_bankend= 'vllm', # 'vllm' or 'huggingface' (exp)
    alias='v0',
)

if __name__ == "__main__":
    # Run the evaluation
    run = wandb.init(project=config.wandb_project, 
                     entity=config.wandb_entity, 
                     job_type="evaluate", 
                     config=config,
                     name=f"eval_vi_llm_{datetime.datetime.now().strftime('%Y-%m-%d')}",
                     )
    
    # Update the config with the wandb run
    config = wandb.config
    evaluate(
        base_model_name=config.base_model,
        wandb_model_name=config.wandb_model,
        data_version=config.data_version,
        model_version=config.alias,
        llm_bankend=config.llm_bankend,
    )
    run.finish()
    print("Evaluation completed.")