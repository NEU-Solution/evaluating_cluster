import json
import os
import re

choices = ['A', 'B', 'C', 'D', 'E', 'F', 'a', 'b', 'c', 'd', 'e', 'f']

def append_jsonl_to_file(json_obj: list|dict, file_path: str):
    with open(file_path, 'a') as f:
        json.dump(json_obj, f)
        f.write('\n')


def get_avaliable_questions(input_path, reference_paths = [], max_questions = 2000):
    selected_questions = []
    done_ids = set()
    
    # Read reference file and get all done ids
    if isinstance(reference_paths, str):
        path = reference_paths.split(',')
        reference_paths = path
        
    for reference_path in reference_paths:
        if os.path.exists(reference_path):
            print(f"Reading reference file: {reference_path}")

            if reference_path.endswith('.jsonl'):
                with open(reference_path, 'r') as f:
                    for line in f:
                        try:
                            msg_obj = json.loads(line)
                            done_ids.add(msg_obj['ids'])
                        except:
                            print(f"Error reading line:")
                            
            elif reference_path.endswith('.json'):
                with open(reference_path, 'r') as f:
                    questions = json.load(f)
                    for question in questions:
                        done_ids.add(question['ids'])


    # Read input file and get all questions
    if input_path.endswith('.jsonl'): 
        print(f"Reading JSONL input file: {input_path}")
        with open(input_path, 'r') as f:
            for line in f:
                question = json.loads(line)
                if question['ids'] not in done_ids:
                    selected_questions.append(question)

    elif input_path.endswith('.json'):
        print(f"Reading JSON input file: {input_path}")
        with open(input_path, 'r') as f:
            questions = json.load(f)
            for question in questions:
                if question['ids'] not in done_ids:
                    selected_questions.append(question)
    else:
        raise ValueError("Input file must be json or jsonl")
                                     

    print(f"Total questions: {len(selected_questions)}")
    return selected_questions


def create_mcq_text(mcq):

    options = ["A", "B", "C", "D", "E"]
    
    question = mcq['question']

    text_choices = ""

    # Check if A B C D is in the answer or not
    flag_a = True 
    match = re.search(r'\b[A-D]\b', mcq['choices'][0])
    if match:
        flag_a = False
        
    
    for i, choice in enumerate(mcq['choices']):
        if flag_a:
            text_choices += f"{options[i]}. {choice}\n"
        else:
            text_choices += f"{choice}\n"



    if isinstance(mcq['answer'], int):
        answer = choices[mcq['answer']]
    else:
        answer = mcq['answer'].upper()

    return {
        'id': mcq['id'],
        'mcq_question': question,
        'choice': text_choices,
        'answer': answer,
    }



# https://github.com/THUDM/ReST-MCTS/blob/main/utils/answer_extractor.py
def extract(answer, q_type = 'MCQ'):
    if '\\boxed{' in answer:
        trunc_ans = answer.split('\\boxed{')[-1]
        extracted_ans = trunc_ans.split('}')[0].strip().replace(' ', '').replace(',', '')
        if '\\textbf{' in extracted_ans:
            extracted_ans = extracted_ans.split('\\textbf{')[-1]
        flag = 1
        if q_type == 'MCQ':
            if len(extracted_ans) == 1 and extracted_ans in choices:
                flag = 1
                extracted_ans = extracted_ans.upper()
            else:
                flag = 0
        elif q_type == 'MCQ(multiple)':
            for let in extracted_ans:
                if let not in choices:
                    flag = 0
                    break
            extracted_ans = extracted_ans.upper()
        else:
            try:
                float_ans = float(extracted_ans)
            except Exception as e:
                flag = 0
        if flag == 1:
            return extracted_ans
        else:
            return 'None'
    elif 'Đáp án' in answer:

        pattern = r'Đáp án\s*:'

        answer = re.split(pattern, answer)[-1]
        answer = answer.strip().upper().replace(' ', '').replace(',', '')[0]
        if len(answer) == 1 and answer in choices:
            return answer
        else:
            return 'None'  
    else:
        answer = answer.strip().upper().replace(' ', '').replace(',', '').replace('AND', '').replace(':', '')
        # print(f'Processed strings:{answer}\n')
        match1 = re.findall(r'[\[,\{,\(][A-D]+[\],\},\)]', answer)
        match2 = re.findall(r'[\[,\{,\(]-?[0-9]+\.?[0-9]*[\],\},\)]', answer)
        match3 = re.findall(r'ANSWERIS-?[0-9]+\.?[0-9]*', answer)
        match4 = re.findall(r'ANSWERIS[A-D]{1,4}', answer)
        match5 = re.findall(r'ANSWER-?[0-9]+\.?[0-9]*', answer)
        match6 = re.findall(r'ANSWER[A-D]{1,4}', answer)
        match7 = re.findall(
            r'[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
            answer)
        match8 = re.findall(r'[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]', answer)
        match9 = re.findall(r'[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]', answer)
        match10 = re.findall(r'ANSWERIS[\[,\{,\(]-?[0-9]+\.?[0-9]*[\],\},\)]', answer)
        match11 = re.findall(r'ANSWER[\[,\{,\(]-?[0-9]+\.?[0-9]*[\],\},\)]', answer)
        match12 = re.findall(r'ANSWERIS[\[,\{,\(][A-D]{1,4}[\],\},\)]', answer)
        match13 = re.findall(r'ANSWER[\[,\{,\(][A-D]{1,4}[\],\},\)]', answer)
        match14 = re.findall(
            r'ANSWERIS[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
            answer)
        match15 = re.findall(r'ANSWERIS[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
                             answer)
        match16 = re.findall(r'ANSWERIS[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]', answer)
        match17 = re.findall(
            r'ANSWER[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
            answer)
        match18 = re.findall(r'ANSWER[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
                             answer)
        match19 = re.findall(r'ANSWER[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]', answer)

        if match14:
            print('Answer matching type 14\n')
            ans = match14[-1]
            ans = ans[8:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match15:
            print('Answer matching type 15\n')
            ans = match15[-1]
            ans = ans[8:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match16:
            print('Answer matching type 16\n')
            ans = match16[-1]
            ans = ans[8:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match12:
            print('Answer matching type 12\n')
            ans = match12[-1]
            ans = ans[8:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match17:
            print('Answer matching type 17\n')
            ans = match17[-1]
            ans = ans[6:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match18:
            print('Answer matching type 18\n')
            ans = match18[-1]
            ans = ans[6:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match19:
            print('Answer matching type 19\n')
            ans = match19[-1]
            ans = ans[6:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match13:
            print('Answer matching type 13\n')
            ans = match13[-1]
            ans = ans[6:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match10:
            print('Answer matching type 10\n')
            ans = match10[-1]
            ans = ans[9:]
            ans = ans[:-1]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')

        if match11:
            print('Answer matching type 11\n')
            ans = match11[-1]
            ans = ans[7:]
            ans = ans[:-1]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')

        if match3:
            print('Answer matching type 3\n')
            ans = match3[-1]
            ans = ans[8:]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')

        if match4:
            print('Answer matching type 4\n')
            ans = match4[-1]
            ans = ans[8:]
            if 'MCQ' in q_type:
                return ans

        if match5:
            print('Answer matching type 5\n')
            ans = match5[-1]
            ans = ans[6:]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')

        if match6:
            print('Answer matching type 6\n')
            ans = match6[-1]
            ans = ans[6:]
            if 'MCQ' in q_type:
                return ans

        if match7:
            print('Answer matching type 7\n')
            ans = match7[-1]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match8:
            print('Answer matching type 8\n')
            ans = match8[-1]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match9:
            print('Answer matching type 9\n')
            ans = match9[-1]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match1:
            print('Answer matching type 1\n')
            ans = match1[-1]
            ans = ans[1:]
            ans = ans[:-1]
            if 'MCQ' in q_type:
                return ans

        if match2:
            print('Answer matching type 2\n')
            ans = match2[-1]
            ans = ans[1:]
            ans = ans[:-1]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')
        print('answer invalid!\n')
        return 'None'


if __name__ == "__main__":
    text = "Đáp án: A"
    print(extract(text))