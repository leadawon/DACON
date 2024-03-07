import os
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import itertools
import pickle
from tqdm import tqdm
import csv
import re
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42) # Seed 고정

model_path = "Phind/Phind-CodeLlama-34B-v2"

# initialize the model


model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_path)






# HumanEval helper

def generate_one_completion(prompt: str):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Generate
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_new_tokens=384, do_sample=True, top_p=0.75, top_k=40, temperature=0.1)
    completion = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    completion = completion.replace(prompt, "").split("\n\n\n")[0]

    return completion

# perform HumanEval
inputtexts = """
### System Prompt
You are an intelligent programming assistant.

### User Message
Implement a linked list in C++

### Assistant
...
"""

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(inputtexts))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]
print(samples)