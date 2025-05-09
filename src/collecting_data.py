import os
import json
import pandas as pd
import sys

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)  

sys.path.append('..')

from src.utils import create_mcq_text

current_dir = os.path.dirname(os.path.abspath(__file__))


def convert_and_save_mcq(data, output_path) -> None:
    mcq_data = []
    for item in data:
        mcq = create_mcq_text(item)
        mcq_data.append(mcq)
    
    with open(output_path, 'w') as f:
        for item in mcq_data:
            f.write(json.dumps(item) + '\n')

    logging.info(f"Saved VMLU to {output_path}")


def dev_load_vmlu(**kwargs) -> None:

    file_path = os.path.join(current_dir, '../example', 'dev_vmlu.jsonl')

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception as e:
                logging.error(f"Error reading line: {line}: {e}")
    logging.info(f"Loaded {len(data)} records from {file_path}. Not saving")

    # Save to jsonl
    output_file_name = '../temp/vmlu.jsonl'
    output_folder = os.path.join(current_dir, '../temp')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file_name)
    convert_and_save_mcq(data, output_path)
    

def dev_load_mmlu(**kwargs) -> None:

    file_path = os.path.join(current_dir, '../example', 'dev_mmlu_vi.json')

    data = []
    with open(file_path, 'r') as f:
        data = json.load(f)

    for i, item in enumerate(data):
        data[i]['id'] = f"mmlu-{i}"

    logging.info(f"Loaded {len(data)} records from {file_path}. Not saving")

    # Save to jsonl
    output_file_name = '../temp/mmlu_vi.jsonl'
    output_folder = os.path.join(current_dir, '../temp')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file_name)
    convert_and_save_mcq(data, output_path)


def dev_load_m3exam(**kwargs) -> None:

    file_path = os.path.join(current_dir, '../example', 'dev_m3exam_vi.json')

    data = []
    with open(file_path, 'r') as f:
        data = json.load(f)

    for i, item in enumerate(data):
        data[i]['id'] = f"m3exam-{i}"

    logging.info(f"Loaded {len(data)} records from {file_path}")

    # Save to jsonl
    # output_file_name = '../temp/m3exam_vi.jsonl'
    # output_folder = os.path.join(current_dir, '../temp')
    # os.makedirs(output_folder, exist_ok=True)
    # output_path = os.path.join(output_folder, output_file_name)

    # convert_and_save_mcq(data, output_path)


def fake_etl(**kwargs) -> None:
    """
    Fake ETL function to simulate data loading and processing.
    """
    dev_load_vmlu()
    dev_load_mmlu()
    # dev_load_m3exam()

def etl():
    pass

if __name__ == "__main__":
    fake_etl()