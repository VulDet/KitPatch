# KitPatch: Knowledge-Augmented Reasoning Paths for Enhanced Vulnerability Patch Generation

## Overview
In this repository, you will find a Python implementation of our KitPatch. KitPatch is a novel automated approach that leverages knowledge-augmented reasoning paths to enhance LLM-based vulnerability patch generation. It consists of two modules: stepwise fix-reasoning path collection and knowledge-augmented patch generation.

## Setting up the environment
You can set up the environment by following commands.
```
conda create -n kitpatch python=3.8.5
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.46.3
pip install openai==1.84.0
pip install scikit-learn
pip install tree-sitter
pip install tree-sitter-c
```

## Package Structure
```
├── Code
    ├── KitPatch
        ├── code_embedding
            ├── utils
                ├── __init__.py
                ├── early_stopping.py
            ├── model.py
            ├── run.py
        ├── graph_embedding
            ├── gat.py
            ├── loss.py
            ├── train.py
            ├── utils.py
        ├── patch generation
            ├── data_preprocess.py
            ├── model.py
            ├── repair_direct_reasoning.py
            ├── repair_kitpatch.py
        ├── VulKG_construction
            ├── import
                ├── bigvul
                    ├── ...
                ├── reposvul
                    ├── ...
            ├── VulKG_Deployment.cypher
├── Datasets
    ├── graph_data
        ├── bigvul_graph_data.pkl 
        ├── reposvul_graph_data.pkl 
    ├── graph_embedding
        ├── bigvul_graph_embeddings.pkl
        ├── reposvul_graph_embeddings.pkl
    ├── prompt
        ├── generate_prompt.txt
        ├── repair_prompt.txt
    ├── source_code
        ├── bigvul
            ├── ...
        ├── reposvul
            ├── ...
   
```

## How to use
Example usage to run repair_kitpatch:

n_example: Number of generated candidate patches.

sample: The maximum number of FiRP generation attempts.

api_key: Place your deepseek access key.
```
python repair_kitpatch.py --dataset bigvul --n_example 10 --sample 25 --model_dir ./saved_models
```
