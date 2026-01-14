import json
import os
import difflib
import pandas as pd


def generate_diff_version(buggy_func, ground_patch_func):
    pre_statements = buggy_func.split('\n')
    post_statements = ground_patch_func.split('\n')

    vul_funcs = []
    patch_funcs = []
    for m, code in enumerate(pre_statements):
        vul_funcs.append((m + 1, code))
    for n, code in enumerate(post_statements):
        patch_funcs.append((n + 1, code))
    vuln_dict = dict(vul_funcs)
    fixed_dict = dict(patch_funcs)
    
    all_lines = sorted(set(vuln_dict.keys()) | set(fixed_dict.keys()))
    
    vuln_lines = []
    fixed_lines = []
    for line_num in all_lines:
        vuln_lines.append(vuln_dict.get(line_num, ""))
        fixed_lines.append(fixed_dict.get(line_num, ""))
    
    differ = difflib.Differ()
    diff = list(differ.compare(vuln_lines, fixed_lines))
    diff_content = []
    for idx, line in enumerate(diff):
        if (line.startswith('-') or line.startswith('+')) and line.replace('-', '').replace('+', '').strip():
            diff_content.append(line)
    diff_content_new = []
    for line in diff_content:
        if line.startswith('-'):
            diff_content_new.append(line)
    for line in diff_content:
        if line.startswith('+'):
            diff_content_new.append(line)
    diff_content_new = "\n".join(diff_content_new)

    return diff_content_new


def clean_parse_bigvul(dataset, folder):
    data_source = pd.read_json(os.path.join(folder+f"{dataset}_data.json"))
    cleaned_result = {}
    repeated_files = []
    for i, row in data_source.iterrows():
        filename = row['vul_filepath'].split('/')[-1] + '--' + row['function_name'] + '.c'
        diff_version = generate_diff_version(row['vul_func_code'], row['patch_func_code'])
        if len(diff_version.split(" ")) > 100 or len(row['vul_func_code'].split(" ")) > 1000:
            continue
        if filename in cleaned_result.keys():
            repeated_files.append(filename)
        cleaned_result[filename] = {"vul": row['vul_func_code']}
        cleaned_result[filename]["diff"] = diff_version
        cleaned_result[filename]["fix"] = row['patch_func_code']
        cleaned_result[filename]["cwe_id"] = row['vul_type']
        cleaned_result[filename]["location"] = row['raw_before_change_lines'] # 漏洞行号

    repeated_files = list(set(repeated_files))
    for file in repeated_files:
        del cleaned_result[file]
    
    return cleaned_result