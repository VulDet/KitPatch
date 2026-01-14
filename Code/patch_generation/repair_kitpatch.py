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
import pickle
import numpy as np
import pandas as pd
from model import Model
from openai import OpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm

from data_preprocess import clean_parse_bigvul

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

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


def remove_here_sentences_multiline(text):
    pattern = r'Here.*?:'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def remove_empty_lines(text):
    # 按行分割，过滤掉空行，再重新组合
    return '\n'.join([line for line in text.splitlines() if line.strip() != ''])


def run_validation(output, ground_patch_func):
    pred_patch = output
    firp = ''
    try:
        pred_patch = pred_patch.split("```")[1].strip("c\n")   
    except:
        pass
    
    pred_patch = remove_c_comments(pred_patch)
    ground_patch_func = remove_c_comments(ground_patch_func)

    prediction_result = pred_patch.split('\n')
    new_prediction = []
    for res in prediction_result:
        res = res.strip()
        if res.startswith('-') and not res.startswith('---') and not res.endswith('.c') and not res.endswith('.h>') and "#include" not in res:
            res = res.strip('-').strip()
            if res.startswith('//') or res.endswith('*/') or len(res)==0:
                continue
            elif '//' in res:
                res = res.split('//')[0]
            res = '-' + ' ' + res
            new_prediction.append(res)
    for res in prediction_result:
        res = res.strip()
        if res.startswith('+') and not res.startswith('+++') and not res.endswith('.c') and not res.endswith('.h>') and "#include" not in res:
            res = res.strip('+').strip()
            if res.startswith('//') or res.endswith('*/') or len(res)==0:
                continue
            elif '//' in res:
                res = res.split('//')[0]
            res = '+' + ' ' + res
            new_prediction.append(res)
    new_prediction = ' '.join(new_prediction)
    new_prediction = ' '.join(new_prediction.split())
    ground_patch_func = ' '.join(ground_patch_func.split())
    new_prediction = clean_tokens(new_prediction)
    ground_patch_func = clean_tokens(ground_patch_func)
    valid = ems(new_prediction, ground_patch_func)

    try:
        firp = output.split("```")[0] + output.split("```")[2]
        firp = remove_here_sentences_multiline(firp)
        firp = remove_empty_lines(firp)
    except:
        pass

    return valid, firp, new_prediction, ground_patch_func


def request_engine(messages):
    ret = None
    while ret is None:
        try:
            ret = client.chat.completions.create(model="deepseek-chat", messages=messages, stream=False)
        except Exception as e:
            print(e)
            return None
    return ret


def extract_all_context_files(type, args):
    context_slice_path = f"./Datasets/{args.dataset}/vul-fix_context/"  
    if type != "test":
        context_slice_path = f"./Datasets/{args.dataset}/vul-fix_context/"   
    else:
        context_slice_path = f"./Datasets/{args.dataset}/vul_context/"
    context_files = {}
    for fi in os.listdir(context_slice_path):
        if args.dataset == "bigvul":
            fi_ = "--".join(fi.split("--")[2:]).split("_slice_")[0]
            if not fi_.startswith('CVE-'):
                continue
        else:
            fi_ = "--".join(fi.split("--")[1:]).split("_slice_")[0]
        
        if fi_ not in context_files:
            context_files[fi_] = [fi]
        else:
            context_files[fi_].append(fi)

    return context_files, context_slice_path

def extract_context_code(filename, context_files, context_slice_path):
    context_file = context_files[filename]
    context_code = ''
    for index, f in enumerate(context_file):
        with open(os.path.join(context_slice_path, f)) as a:
            content = a.read()
        context_code = context_code + f'\n//Context Code {str(index+1)}:\n' + content

    return context_code


def get_embedding(args, model, tokenizer, think_dataset):
    with open(f"./Results/FiRP/{args.dataset}/FiRP.json", 'r') as f:
        firp = json.load(f)
    for file, repair_result in firp.items():
        buggy = think_dataset[file]["vul"]
        tokenized_code = tokenizer.encode_plus(buggy, max_length=400, return_tensors="pt")
        outputs = model(**tokenized_code)    
        firp[file]["embedding"] = outputs[0][0, 0, :].detach().numpy().tolist()
    with open(f"./Results/FiRP/{args.dataset}/FiRP_embedding.json", 'w') as f:
        json.dump(firp, f, indent=4)


def get_clusters(args, think_dataset):
    firp = json.load(open(f"./Results/FiRP/{args.dataset}/FiRP_embedding.json", 'r'))
    firp_think = {}
    for file in think_dataset.keys():
        if file in firp:
            firp_think[file] = firp[file]
    embeddings = np.asarray([repair_result["embedding"] for _, repair_result in firp_think.items()]) 
    kmeans = KMeans(n_clusters=args.n_example, n_init=10, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels
    

def get_example_from_clusters(args, model, tokenizer, buggy, labels, file):
    with open(f"./Datasets/graph_embedding/{args.dataset}_graph_embeddings.pkl", "rb") as f:
        graph_data = pickle.load(f)
    
    firp = json.load(open(f"./Results/FiRP/{args.dataset}/FiRP_embedding.json", 'r'))
    embeddings = []
    for _, repair_result in firp.items():
        try:
            emb_ = repair_result["embedding"] + graph_data['-'.join(_.split('-')[:3])]
        except:
            emb_ = repair_result["embedding"] + [float(0)]*128
        embeddings.append(emb_)
    embeddings = np.asarray(embeddings)
    
    points = [file for file, repair_result in firp.items()]

    selected_points = []
    
    tokenized_code = tokenizer.encode_plus(buggy, max_length=400, return_tensors="pt")
    outputs = model(**tokenized_code)    
    buggy_embedding = outputs[0][0, 0, :].detach().numpy().tolist()
    try:
        buggy_embedding = buggy_embedding + graph_data['-'.join(file.split('-')[:3])]
    except:
        buggy_embedding = buggy_embedding + [float(0)]*128

    for i in range(args.n_example):
        cluster_points = np.where(labels == i)[0]
        cluster_embeddings = [embeddings[point] for point in cluster_points]
        similarities = cosine_similarity([buggy_embedding], cluster_embeddings)
        most_similar_index = np.argmax(similarities)
        selected_point = cluster_points[most_similar_index]
        selected_points.append(points[selected_point])
    return selected_points


def add_bug_comments(code_string, buggy_line_numbers):
    lines = code_string.split("\n")
    for line_number in buggy_line_numbers:
        if 1 <= line_number <= len(lines):
            lines[line_number-1] += "//Vulnerable line"   
    modified_code_string = "\n".join(lines)
    return modified_code_string


def extract_kg_context_from_json(filename, test=False):
    CVE_ID = '-'.join(filename.split('-')[:3])
    with open(os.path.join("./Datasets/VulKG/bigvul", CVE_ID+".json"), "r", encoding="utf-8") as f:
        VulKG = json.load(f)
    new_VulKG = []
    if test:
        for item in VulKG:
            if item["source_node"]["labels"][0]=="Commit" or item["target_node"]["labels"][0]=="Commit":
                continue
            new_VulKG.append(item)
        return new_VulKG
    else:
        return VulKG


def remove_c_comments(code):
    """
    移除C/C++代码中的注释，同时保留字符串内容
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # 注释替换为空格保持行号不变
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    cleaned_code = re.sub(pattern, replacer, code)
    # 移除连续的空白行
    cleaned_code = '\n'.join(
        line for line in cleaned_code.split('\n')
        if line.strip() != ''
    )

    return cleaned_code


def generate_reasoning_path(args, think_dataset):
    with open(f"./Datasets/prompt/generate_prompt.txt") as f:
        generate_prompt = f.read()
    print(generate_prompt)
    try:
        with open(f"./Results/FiRP/{args.dataset}/FiRP.json", 'r') as f:
            firp = json.load(f)
    except:
        firp = {}

    context_slice_path = f"./Datasets/vul-fix_context/{args.dataset}/"
    context_files = {}
    for fi in os.listdir(context_slice_path):
        fi_ = "--".join(fi.split("--")[2:]).split("_slice_")[0]
        if fi_ not in context_files:
            context_files[fi_] = [fi]
        else:
            context_files[fi_].append(fi)

    i = 0
    for file, bug in tqdm(think_dataset.items()):
        print(i)
        i += 1
        if file in firp.keys(): continue
        print("Repairing bug {} ... ".format(file.split(".")[0]))
        try:
            VulKG = extract_kg_context_from_json(file)
        except:
            VulKG = " "
        
        buggy_func = bug["vul"]
        ground_patch_func = bug["diff"]
        buggy_lines = bug["location"]
        modified_func = add_bug_comments(buggy_func, buggy_lines)
        vul_lines = ""
        for line in bug['diff'].splitlines():
            if line.startswith("-"):
                vul_lines += line[1:] + "\n"

        try:
            language = bug["language"]
        except:
            language = "C/C++"
        if file in context_files:
            context_file = context_files[file]
            context_code = ''
            for index, f in enumerate(context_file):
                with open(os.path.join(context_slice_path, f)) as a:
                    content = a.read()
                context_code = context_code + f'\n//Context Code {str(index+1)}:\n' + content
        else:
            context_code = ''
     
        prompt = generate_prompt.format(language=language, vulFunction=modified_func, contextCode=context_code, vul_line=vul_lines, VulKG=VulKG)  

        for _ in range(args.sample):
            print(file + '---' + str(_))
            messages = [
                        {"role": "system", "content": "You are an Automatic Vulnerability Repair Tool"},
                        {"role": "user", "content": prompt}
                    ]
            output = request_engine(messages).choices[0].message.content
            valid, pred_firp, _, _ = run_validation(output, ground_patch_func)
            if valid:
                break
        
        firp[file] = {"output": pred_firp, "valid": valid}
        with open(f"./Results/FiRP/{args.dataset}/FiRP.json", 'w') as f:
            json.dump(firp, f, indent=4)

    

def generate_repair_patches(args, Select_model, Select_tokenizer, think_dataset, inference_files, output_filename="SynEq"):
    with open(f"./Datasets/prompt/repair_prompt.txt") as f:
        repair_prompt = f.read()
    with open(f"./Results/FiRP/{args.dataset}/FiRP_embedding.json", 'r') as f:
        firp = json.load(f)

    try:
        with open(f"./Results/repair_result/{args.dataset}_{output_filename}.json", 'r') as f:
            repair = json.load(f)
    except:
        repair = {}

    labels = get_clusters(args, think_dataset)


    test_context_files, test_context_slice_path = extract_all_context_files("test", args)
    example_context_files, example_context_slice_path = extract_all_context_files("FiRP", args)

    for file, bug in tqdm(inference_files.items()):
        print("Repairing vulnerability {} ... ".format(file.split(".")[0]))
        examples = get_example_from_clusters(args, Select_model, Select_tokenizer, bug['vul'], labels, file)
        
        if file in repair.keys():
            repair_results = repair[file]
        else:
            repair_results = []
        
        try:
            vul_code = extract_context_code(file, test_context_files, test_context_slice_path)  
        except:
            vul_code = add_bug_comments(bug["vul"], bug["location"])

        vul_lines = ""
        for line in bug['diff'].splitlines():
            if line.startswith("-"):
                vul_lines += line[1:] + "\n"

        try:
            VulKG = extract_kg_context_from_json(file, test=True)
        except:
            VulKG = " "
        try:
            language = bug['language']
        except:
            language = 'c/c++'
    
        for i in range(len(examples)):
            try:
                example_vul = extract_context_code(examples[i], example_context_files, example_context_slice_path)
            except:
                example_vul = add_bug_comments(think_dataset[examples[i]]["vul"], think_dataset[examples[i]]["location"])
            
            example_vul_lines = ""
            for line in think_dataset[examples[i]]['diff'].splitlines():
                if line.startswith("-"):
                    example_vul_lines += line[1:] + "\n"

            instruct_1 = "\n\n//Here is the diff:\n\n```%s\n```" % think_dataset[examples[i]]["diff"]
            
            prompt = repair_prompt.format(language=language, example_vul=example_vul, example_vul_line=example_vul_lines, example_FiRP=firp[examples[i]]["output"]+instruct_1, vul=vul_code, vul_line=vul_lines, VulKG=VulKG) 
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
            # print(pred_patch)
            
            repair_result.append({"output": output, "valid": valid, "pred_patch": pred_patch, "ground_patch": ground_patch})

            repair_results.append(repair_result)
            repair[file] = repair_results
            with open(f"./Results/repair_result/{args.dataset}_{output_filename}.json", 'w') as f:
                json.dump(repair, f, indent=4)


def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path) 
    model = Model(model, config, tokenizer, args)
    model.to(device)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  

    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    model_dir = os.path.join(args.model_dir, '{}'.format(checkpoint_prefix))  
    model = torch.load(model_dir)
    
    return model, tokenizer    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bigvul",
                        help="Dataset to use")
    parser.add_argument("--n_example", type=int, default=10)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--chance", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # model
    parser.add_argument("--model_dir", default="saved_models", type=str)
    parser.add_argument("--model_name_or_path", default="unixcoder-base-nine", type=str, 
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization.")
    
    args = parser.parse_args()

    out_folder = f'./Results/FiRP/{args.dataset}'
    os.makedirs(out_folder, exist_ok=True)

    my_dataset = clean_parse_bigvul(args.dataset, "./Datasets/")
    set_seed(args.seed)

    total_files = list(my_dataset.keys())
    num_test = int(len(total_files) * 0.2)
    test_files = random.choices(total_files, k=num_test)
    think_dataset = {key: value for key, value in my_dataset.items() if key not in test_files}   
    inference_dataset = {key: value for key, value in my_dataset.items() if key in test_files}

    generate_reasoning_path(args, think_dataset)
    
    Select_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    Select_model = AutoModel.from_pretrained(args.model_dir)
    get_embedding(args, Select_model, Select_tokenizer, think_dataset)

    generate_repair_patches(args, Select_model, Select_tokenizer, think_dataset, inference_dataset)



if __name__ =="__main__":
    main()
