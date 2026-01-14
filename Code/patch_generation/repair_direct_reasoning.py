from __future__ import absolute_import, division, print_function
import os
import time
import json
import torch
import random
import argparse
import tiktoken
import warnings
import re
import string
import regex
import numpy as np
from model import Model
from openai import OpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm

from repair_kitpatch import run_validation
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

encoding_name = 'cl100k_base'
encoding = tiktoken.get_encoding(encoding_name)

client = OpenAI(api_key="xx", base_url="https://api.deepseek.com")



def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    normal_prediction = normalize_answer(prediction)
    normal_groundtruth = normalize_answer(ground_truth)
    if normal_groundtruth == normal_prediction:
        return True
    return False


def ems(prediction, ground_truth):
    return exact_match_score(prediction, ground_truth)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def chatgpt_encoding_count(input: str) -> int:
    """ chatgpt token count """
    token_integers = encoding.encode(input)
    num_tokens = len(token_integers)
    return num_tokens


def remove_here_sentences_multiline(text):
    pattern = r'Here.*?:'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def remove_empty_lines(text):
    # 按行分割，过滤掉空行，再重新组合
    return '\n'.join([line for line in text.splitlines() if line.strip() != ''])


def request_engine(messages):
    ret = None
    while ret is None:
        try:
            ret = client.chat.completions.create(model="deepseek-chat", messages=messages,
                                                 stream=False) 
        except Exception as e:
            print(e)
            return None
    return ret


def add_bug_comments(code_string, buggy_line_numbers):
    lines = code_string.split("\n")
    for line_number in buggy_line_numbers:
        if 1 <= line_number <= len(lines):
            lines[line_number - 1] += " // Vulnerable Line"
    modified_code_string = "\n".join(lines)
    return modified_code_string


def remove_c_comments(code):
    """
    移除C/C++代码中的注释，同时保留字符串内容
    """

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    cleaned_code = re.sub(pattern, replacer, code)
   
    cleaned_code = '\n'.join(
        line for line in cleaned_code.split('\n')
        if line.strip() != ''
    )

    return cleaned_code


def generate_patch(dataset, inference_files, output_filename="deepseek_v3_base_SynEq"):
    repair_prompt = """// Provide a diff to fix the vulnerability in the given c/c++ code.
// Vulnerable Function
{bug}"""
    try:
        with open(f"./Results/repair_result/{dataset}_{output_filename}.json", 'r') as f:
            repair = json.load(f)
    except:
        repair = {}

    inference_files_new = {}
    for key, value in inference_files.items():
        if key not in repair.keys():
            inference_files_new[key] = value

    for file, bug in tqdm(inference_files_new.items()):
        print("Repairing bug {} ... ".format(file.split(".")[0]))

        if file in repair.keys():
            repair_results = repair[file]
        else:
            repair_results = []
        
        buggy = add_bug_comments(bug["vul"], bug["location"])
        for i in range(10-len(repair_results)):
            prompt = repair_prompt.format(bug=buggy)    
            ground_patch_func = bug["diff"]

            messages = [
                        {"role": "system", "content": "You are an Automatic Vulnerability Repair Tool."},
                        {"role": "user", "content": prompt}
                    ]
            repair_result = []
            try:
                output = request_engine(messages).choices[0].message.content
            except:
                continue
            valid, message, pred_patch, ground_patch = run_validation(output, ground_patch_func)
            print(pred_patch)
            
            repair_result.append({"output": output, "valid": valid, "pred_patch": pred_patch, "ground_patch": ground_patch})
            if valid: 
                break

            repair_results.append(repair_result)
            repair[file] = repair_results
            with open(f"./Results/repair_result/{dataset}_{output_filename}.json", 'w') as f:
                json.dump(repair, f, indent=4)
            
            if valid:
                break


def main():
    dataset = "bigvul"

    with open(f"./Results/FiRP/bigvul/inference_dataset.json", 'r') as f:
        inference_dataset = json.load(f)

    generate_patch(dataset, inference_dataset)


if __name__ == "__main__":
    main()