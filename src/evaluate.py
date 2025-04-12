import sys 
import os
import json
sys.path.append('..')
import wandb    

current_dir = os.path.dirname(os.path.abspath(__file__))

from src.collecting_data import fake_etl
from src.scoring import evaluate_generation
from src.load_model import (download_model_regristry, 
                            start_inference_server, 
                            terminate_server, 
                            # load_huggingface_model
                            )
from src.logging import BaseLogger, create_logger

from llm import OpenAIWrapper, LLM

import logging
import pandas as pd


def log_result(logger: BaseLogger, results: list[dict], dataset_name: str) -> float:

    score = 0
    for result in results:
        score += result['score']
    
    avg_score = score/len(results) * 100

    logger.log_metric(dataset_name, avg_score)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(current_dir, '../data', f"{dataset_name}.csv"), index=False)

    # table = wandb.Table(dataframe=df)
    logger.log_table(f"{dataset_name}_response", df)

    return avg_score



def llm_evaluate(llm: LLM, current_dir: str, logger: BaseLogger = None, multi_thread:bool = True, max_workers:int = 2) -> list[dict]:
    """
    Evaluate the model using the provided questions and answers.
    """
    # Scan for all jsonl files in the data folder
    data_folder = os.path.join(current_dir, '../temp')
    question_paths = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.jsonl'):
                question_paths.append(os.path.join(root, file))


    logging.info(f'Found {len(question_paths)} jsonl files in the data folder.')
    # Evaluate each jsonl file

    evaluate_results = dict()

    for question_path in question_paths:
        
        eval_dataset_name = os.path.basename(question_path).replace('.jsonl', '')
        logging.info(f"Evaluating {eval_dataset_name}...")

        results = evaluate_generation(llm, question_path, multi_thread=multi_thread, max_workers=max_workers)
        evaluate_results[eval_dataset_name] = results

        # Log the results
        if logger is not None:
            score = log_result(logger, results, eval_dataset_name)
            logger.update_summary(f"{eval_dataset_name}_final_score", score)

        logging.info(f"Evaluation completed for {eval_dataset_name}.")


    

def evaluate(base_model_name: str, lora_name: str, data_version: str, logger = None, model_version: str = None, multi_thread:bool = True, llm_bankend = 'vllm', max_workers:int = 2, port:int = 8000, tracking_backend: str = 'wandb'):

    fake_etl()
    
    inner_run = False
    if logger is None:
        inner_run = True
        # Initialize a new logger with a new run
        logger = create_logger(tracking_backend)
        logger.login()
        logger.init_run(
            project=os.getenv("WANDB_PROJECT") if tracking_backend == 'wandb' else os.getenv("MLFLOW_EXPERIMENT_NAME", "Default"),
            entity=os.getenv("WANDB_ENTITY") if tracking_backend == 'wandb' else None,
            job_type="evaluate"
        )

    lora_path = download_model_regristry(lora_name, version=model_version,logger=logger)

    if llm_bankend == 'vllm':
        # Start the inference server
        server_process = start_inference_server(base_model_name, lora_path, port=port, max_vram = 12)
        if server_process is None:
            print("Failed to start the inference server.")
            return

        try:
            host = f"http://localhost:{port}/v1"
            model_name = "evaluate"
            api_key = 'ngu'

            llm = OpenAIWrapper(model_name=model_name, host=host, api_key=api_key)

            llm_evaluate(
                llm,
                current_dir=current_dir,
                logger=logger,
                multi_thread=multi_thread,
                max_workers=max_workers
            )
            
        except Exception as e:
            logging.error(f"An error occurred during evaluation: {str(e)}")
            raise
        finally:
            # Terminate the server process
            terminate_server(server_process)
            logging.info("Inference server terminated.")

            
    else:
        raise ValueError(f"Unknown or not implemented llm backend: {llm_bankend}")
        # llm = load_huggingface_model(base_model_name, lora_path)
        # llm_evaluate(
        #     llm,
        #     current_dir=current_dir,
        #     run=run,
        #     multi_thread=multi_thread,
        #     max_workers=max_workers
        # )
        
    if inner_run:
        # Finish the W&B run
        logger.finish_run()
        logging.info("Tracking run finished.")




if __name__ == "__main__":
    base_model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    lora_name = 'wandb-registry-model/initial-sft'

    evaluate(
        base_model_name=base_model_name,
        lora_name=lora_name,
        data_version='v0.1',
        tracking_backend = 'wandb',
    )