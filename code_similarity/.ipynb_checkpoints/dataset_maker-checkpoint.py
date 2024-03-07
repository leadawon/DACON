import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import os
import random
import numpy as np
import itertools
import pickle
from tqdm import tqdm
import csv

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42) # Seed 고정

MODEL_NAME = "neulab/codebert-cpp"
MODEL_TAG = "neulab_codebert-cpp"
root_dir = "./bigdata/train_code" 




class CodePairsDataset(Dataset):
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

    def create_csv_dataset(self, root_dir, csv_file_path):
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["file_path1", "file_path2", "label"])  # CSV 헤더

            problem_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            problem_files_count = {d: len([f for f in os.listdir(os.path.join(root_dir, d)) if f.endswith('.cpp')]) for d in problem_dirs}

            # 긍정적 쌍 생성
            for problem_dir, file_count in tqdm(problem_files_count.items(), desc="Creating positive pairs"):
                for i in range(1, file_count + 1):
                    for j in range(i + 1, file_count + 1):
                        file1 = os.path.join(root_dir, problem_dir, f"{problem_dir}_{i}.cpp")
                        file2 = os.path.join(root_dir, problem_dir, f"{problem_dir}_{j}.cpp")
                        writer.writerow([file1, file2, 1])  # CSV에 쓰기

            # 부정적 쌍 생성
            for problem_dir, file_count in tqdm(problem_files_count.items(), desc="Creating negative pairs"):
                num_neg_samples_per_file = file_count // 2

                for i in range(1, file_count + 1):
                    current_file = f"{problem_dir}_{i}.cpp"
                    neg_samples_added = 0
                    while neg_samples_added < num_neg_samples_per_file:
                        other_dir = random.choice(list(problem_files_count.keys()))
                        if other_dir == problem_dir:
                            continue
                        
                        other_file_index = random.randint(1, problem_files_count[other_dir])
                        other_file = f"{other_dir}_{other_file_index}.cpp"

                        writer.writerow([os.path.join(root_dir, problem_dir, current_file), os.path.join(root_dir, other_dir, other_file), 0])
                        neg_samples_added += 1

    def load_from_csv(self, csv_file_path):
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # 헤더 건너뛰기
            self.samples = [(row[0], row[1], int(row[2])) for row in reader]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path1, file_path2, label = self.samples[idx]
        with open(file_path1, 'r', encoding='utf-8') as f:
            text1 = f.read()
        with open(file_path2, 'r', encoding='utf-8') as f:
            text2 = f.read()

        inputs1 = self.tokenizer(text1, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        inputs2 = self.tokenizer(text2, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        return {
            "input_ids1": inputs1['input_ids'].squeeze(0),
            "attention_mask1": inputs1['attention_mask'].squeeze(0),
            "input_ids2": inputs2['input_ids'].squeeze(0),
            "attention_mask2": inputs2['attention_mask'].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }

# 사용 예시
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = CodePairsDataset(tokenizer)
dataset.create_csv_dataset(root_dir, f'./bigdata/csvs/{MODEL_TAG}_dataset.csv')  # 데이터셋을 CSV 파일로 생성
dataset.load_from_csv(f'./bigdata/csvs/{MODEL_TAG}_dataset.csv')  # 생성된 CSV 파일에서 데이터셋 로드
