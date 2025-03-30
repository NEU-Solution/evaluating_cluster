import sys 
import os
sys.path.append('..')

from src.collecting_data import fake_etl
from src.scoring import evaluate_generation

from llm import RotateGemini

def test_scoring_mcq():
    fake_etl()

    mcq_paths = ['../temp/m3exam_vi.jsonl']
    llm = RotateGemini(model_name='gemini-2.0-flash-lite')

    for mcq_path in mcq_paths:
        results = evaluate_generation(llm, mcq_path, multi_thread=True, max_workers=4)

        print(results)


if __name__ == "__main__":
    test_scoring_mcq()



