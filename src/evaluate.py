import sys 
import os
import json
sys.path.append('..')
import wandb    

current_dir = os.path.dirname(os.path.abspath(__file__))

from src.collecting_data import fake_etl
from src.scoring import evaluate_generation
from src.load_model import download_model_regristry, start_inference_server, terminate_server

from llm import OpenAIWrapper

import logging
import pandas as pd


WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

wandb.login(key = WANDB_API_KEY)


def log_result(results: list[dict], dataset_name: str):

    score = 0
    for result in results:
        score += result['score']

    wandb.log({dataset_name: score/len(results) * 100})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(current_dir, '../data', f"{dataset_name}.csv"), index=False)

    table = wandb.Table(dataframe=df)
    wandb.log({dataset_name: table})


    

def evaluate(base_model_name: str, wandb_model_name: str, data_version: str, model_version: str = None, multi_thread:bool = True, max_workers:int = 2, port:int = 8000):

    fake_etl()

    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="evaluate")

    lora_path = download_model_regristry(wandb_model_name, version=model_version)

    # Start the inference server
    server_process = start_inference_server(base_model_name, lora_path, port=port)
    if server_process is None:
        print("Failed to start the inference server.")
        return

    host = f"localhost:{port}"
    model_name = "evaluate"
    api_key = 'ngu'

    llm = OpenAIWrapper(model_name=model_name, host=host, api_key=api_key, multimodal=True)

    # Scan for all jsonl files in the data folder
    data_folder = os.path.join('..', 'data')
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
        log_result(run, results, question_path)
        logging.info(f"Evaluation completed for {eval_dataset_name}.")
    
    # Terminate the server process
    terminate_server(server_process)
    logging.info("Inference server terminated.")

    # Finish the W&B run
    run.finish()
    logging.info("W&B run finished.")
