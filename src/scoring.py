from llm import RotateGemini, LLM
from llm.llm_utils import get_json_from_text_response

import json
from src.utils import *
from src.collecting_data import fake_etl, etl

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

current_dir = os.path.dirname(os.path.abspath(__file__))

def single_scoring_mcq(llm: LLM, response: dict, output_path: str) -> dict:

    mcq = response['mcq_question']
    choice = response['choice']
    correct_choice = response['answer']

    system_prompt = """
Bạn được cung cấp một câu hỏi trắc nghiệm và các lựa chọn trả lời.
Bạn hãy suy nghĩ từng bước một cách cẩn thận và chọn ra câu trả lời đúng nhất.
Đáp án của bạn phải là một trong các lựa chọn A, B, C, D.

Đáp án lựa chọn của bạn phải được để trong \\boxed{} và viết hoa (ví dụ: \\boxed{A}).
"""

    prompt = f"""
<question>
{mcq}
</question>

<choices>
{choice}
</choices>
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    answer = llm(messages)

    try:
        choice = extract(answer, 'MCQ')

        print(f"Choice: {choice} | Correct: {correct_choice}")

        if choice == correct_choice:
            score = 1
        else:
            score = 0
    except Exception as e:
        print(e)
        score = 0

    result = dict()
    result['id'] = response['id']
    result['score'] = score
    result['question'] = mcq
    result['response'] = answer

    append_jsonl_to_file(result, output_path)

    return result


def scoring_mcq(llm: LLM, questions: list[dict], output_path: str, max_workers: bool = 4, multi_thread: bool = False):
    
    total_questions = len(questions)
    results = []

    if multi_thread:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for question in questions:
                future = executor.submit(single_scoring_mcq, llm, question, output_path)
                futures[future] = question
                
            # Process results as they complete with proper error handling
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                question = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error processing question ID {question.get('id', 'unknown')}: {str(e)}")
                    # Create a failed result entry
                    failed_result = {
                        'id': question.get('id', 'unknown'),
                        'score': 0,
                        'question': question.get('mcq_question', 'unknown'),
                        'response': None
                    }
                    results.append(failed_result)

    else:
        for question in tqdm(questions, desc="Evaluating"):
            results.append(single_scoring_mcq(llm, question, output_path))

    score = 0
    for result in results:
        # try:
            score += result['score']
        # except:
        #     pass

    logging.info(f"===== Total questions: {total_questions} =====")
    logging.info(f"===== Out of responsed answer, Score: {score/total_questions * 100} =====") 

    return results


def evaluate_generation(llm: LLM, question_path : str, multi_thread : bool = False, max_workers:  int = 4) -> list[dict]:
    
    question = []
    with open(question_path, 'r') as f:
        for line in f:
            question.append(json.loads(line))
    
    
    output_file_name = llm.model_name.replace('/', '__') + '-evaluate-' + os.path.basename(question_path)
    
    output_folder = os.path.join(current_dir, '../data')
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, output_file_name)

    print(f"Number of questions: {len(question)}")

    return scoring_mcq(llm, question[:5], output_path, max_workers, multi_thread)


