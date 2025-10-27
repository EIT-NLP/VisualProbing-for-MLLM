import argparse
import json
import re
import time
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

# Create the parser
parser = argparse.ArgumentParser(description='Process OpenAI API key and JSONL file path.')

# Add arguments
parser.add_argument('--openai_api_key', default="sk-xxxxxx", help='Your OpenAI API key')
parser.add_argument('--answer_file', default="/code/chr/MobileVLM_Dense/eval_results/CVBench/answers/mobilevlm_v2_1.7b_20240819_000827/linear-23-bts256-bts128.finetune.jsonl", help='Path to the JSONL file')
parser.add_argument('--output_dir', default="/code/chr/MobileVLM/chr_toolkits/cvbench_gpt/Joeylin", help='Directory to save the updated JSON file')

# Parse arguments
args = parser.parse_args()

client = OpenAI(api_key=args.openai_api_key, base_url="https://api.deepseek.com")
NUM_SECONDS_TO_SLEEP = 2

# Define a function to query the OpenAI API and evaluate the answer
def get_option_answer(question):
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": question},
                ],
                temperature=0,
            )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    answer = response.choices[0].message.content
    
    single_letter_regex = re.compile(r"^[A-Z]$", re.IGNORECASE)
    
    if single_letter_regex.match(answer):
        return answer.upper()
    else:
        return "Could not determine"

# Ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Function to process a single line from the JSONL file
def process_line(line, correct_answer, index):
    data = json.loads(line)
    question, model_response = data["prompt"], data["text"]
    question4gpt = f'You are given a question and an answer. The question has multiple-choice options labeled A, B, C, D, etc. \
    Your task is to determine which option (A, B, C, D, etc.) the provided answer best corresponds to. Please choose the option that most accurately matches the answer. \
    Important Instructions: \n \
    1. Your response must be only a single letter (A, B, C, D, etc.). \
    2. Do not include any additional text, numbers, or punctuation. \
    3. The output should be exactly one character long, corresponding to the correct option. \
    Here is the question: {question}; Here is the answer: {model_response}'
    gpt_grade = get_option_answer(question4gpt)
    
    data['gpt'] = gpt_grade
    return index, data, gpt_grade in correct_answer

# Read and process the JSONL file
with open(args.answer_file, 'r') as file:
    lines = file.readlines()

num_correct, num_total = 0, 0

results = OrderedDict()  # To maintain the order of processing

# Process each line with a progress bar and ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_line, line, json.loads(line)["answer"], i): i for i, line in enumerate(lines)}

    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(args.answer_file)}"):
        index, processed_item, is_correct = future.result()
        results[index] = processed_item  # Store the result with its index

        if is_correct:
            num_correct += 1

        num_total += 1

# Write results to the output file
output_file_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.answer_file))[0] + '_gpt.jsonl')
with open(output_file_path, 'w') as output_file:
    for index in sorted(results):
        output_file.write(json.dumps(results[index], ensure_ascii=False) + '\n')

print(f"The accuracy for {os.path.basename(args.answer_file)} is {num_correct/num_total:.2f}")