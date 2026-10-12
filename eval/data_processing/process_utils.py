import regex
import random
import json

from data_processing.answer_extraction import extract_math_answer, strip_string

def process_gsm8k_test(item):
    sample = {
        'dataset': 'gsm8k-cot',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': item['question']},
            {'role': 'assistant', 'content': regex.sub(r"<<[^<>]*>>", "", item['cot']) + "\nSo the answer is $\\boxed{" + item['answer'].strip() + "}$."}
        ],
        'answer': item['answer'].replace(',', '')
    }
    yield sample

def process_math_test(item):
    question = item["problem"]
    try:
        answer = extract_math_answer(question, item['solution'], task="cot")
    except:
        return
    sample = {
        "dataset": "math-cot",
        "id": item['id'],
        "level": item["level"],
        "type": item["type"],
        "category": item["category"],
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "\n".join(regex.split(r"(?<=\.) (?=[A-Z])", item["solution"]))}
        ],
        "answer": answer
    }
    yield sample

def process_math500(item):
    question = item["problem"]
    try:
        answer = item["answer"]
    except:
        return
    sample = {
        "dataset": "math500-cot",
        # math500/test.jsonl keys rows by 'unique_id'; keep 'id' as fallback
        # for copies of the dataset that use the conventional name.
        "id": item.get('id', item.get('unique_id')),
        "level": item["level"],
        "subject": item["subject"],
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "\n".join(regex.split(r"(?<=\.) (?=[A-Z])", item["solution"]))}
        ],
        "answer": answer
    }
    yield sample

def process_olympiad_bench(item):
    question = item['question']
    answer = item['final_answer']
    sample = {
        'dataset': 'olympiad_bench',
        'id': str(item['id']),
        "category": item['subfield'],
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "answer": answer[0]
    }
    yield sample

def process_amc(item):
    question = item['problem']
    answer = item['answer']
    sample = {
        'dataset': 'amc',
        'id': str(item['id']),
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "answer": str(answer)
    }
    yield sample

def process_math_sat(item):
    options = item['options'].strip()
    assert 'A' == options[0]
    options = '(' + options
    for ch in 'BCDEFG':
        if f' {ch}) ' in options:
            options = regex.sub(f' {ch}\) ', f" ({ch}) ", options)
    question = f"{item['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options.strip()}"
    messages = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': item['Answer']}
    ]
    item = {
        'dataset': 'math_sat',
        'id': item['id'],
        'language': 'en',
        'messages': messages,
        'answer': item['Answer'],
    }
    yield item

def process_ocwcourses(item):
    messages = [
        {'role': 'user', 'content': item['problem'].strip()},
        {'role': 'assistant', 'content': item['solution'].strip()}
    ]
    item = {
        "dataset": "OCWCourses",
        "id": item['id'],
        "language": "en",
        "messages": messages,
        "answer": item['answer']
    }
    yield item

def process_mmlu_stem(item):
    options = item['options']
    for i, (label, option) in enumerate(zip('ABCD', options)):
        options[i] = f"({label}) {str(option).strip()}"
    options = ", ".join(options)
    question = f"{item['question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{options}"
    messages = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': item['answer']}
    ]
    item = {
        "dataset": "MMLU-STEM",
        "id": item['id'],
        "language": "en",
        "messages": messages,
        "answer": item['answer']
    }
    yield item

def process_tabmwp(item):
    title_str = f'regarding "{item["table_title"]}" ' if item['table_title'] else ''
    question = f'Read the following table {title_str}and answer a question:\n'
    question += f'{item["table"]}\n{item["question"]}'
    if item['choices']:
        question += f' Please select from the following options: {item["choices"]}'
    item = {
        'dataset': 'tabmwp',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': ''}
        ],
        'type': 'multiple-choice' if item['choices'] else 'short-answer',
        "answer": item['answer']
    }
    yield item

def process_mathqa(item):
    question = item['problem'].strip()
    question = question[0].upper() + question[1:]
    # turn a ) 5600 , b ) 6000 , c ) 237 , d ) 7200 , e ) 8600
    # to (A) 5600, (B) 6000, (C) 237, (D) 7200, (E) 8600
    options = item['options']
    options = regex.sub(r'([a-z])\s*\)\s*([^,]+)\s*,?\s*', 
                        lambda m: f"({m.group(1).upper()}) {m.group(2).strip()}, ", 
                        options).strip().rstrip(',')
    
    question = f"{question}\nWhat of the following is the right choice? Explain your answer.\n{options}"
    gt_answer = item['correct']
    item = {
        'dataset': 'mathqa',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': ''}
        ],
        "answer": str(gt_answer)
    }
    yield item

def process_svamp(item):
    body = item['Body'].strip()
    if not body.endswith('.'):
        body += '.'
    question = f'{body} {item["Question"].strip()}'
    item = {
        'dataset': 'svamp',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': str(item['Answer'])}
        ],
        "answer": str(item['Answer'])
    }
    yield item

def process_asdiv(item):
    gt_answer = item['answer']
    gt_answer = regex.sub(r"\(.*?\)", "", gt_answer)
    question = f"{item['body'].strip()} {item['question'].strip()}"
    item = {
        'dataset': 'asdiv',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': ''}
        ],
        "answer": gt_answer
    }
    yield item

def process_mawps(item):
    gt_answer = item['target']
    question = item['input']
    item = {
        'dataset': 'mawps',
        'id': item['id'],
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': ''}
        ],
        "answer": str(gt_answer)
    }
    yield item

def process_gpqa(item):
    options = [
        item["Correct Answer"],
        item["Incorrect Answer 1"],
        item["Incorrect Answer 2"],
        item["Incorrect Answer 3"]
    ]
    random.shuffle(options)
    correct_index = options.index(item["Correct Answer"])
    option_dict = {
        'A': options[0],
        'B': options[1],
        'C': options[2],
        'D': options[3]
    }
    option_str = ""
    for k, v in option_dict.items():
        option_str = f"{option_str}({k}) {v}\n"
    
    formatted_item = {
        "dataset": "gpqa",
        "id": item['id'],
        'answer': chr(65 + correct_index),  # 将索引转换为A、B、C、D
        'messages': [
            {'role': 'user', 'content': f"{item['Question'].strip()}\nWhat of the following is the right choice? Explain your answer.\n{option_str.strip()}"},
            {'role': 'assistant', 'content': ''}
        ]
    }
    yield formatted_item
