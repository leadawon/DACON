import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-7b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


import pandas as pd
import re
from tqdm import tqdm

def remove_comments(cpp_code):
        # 멀티라인 주석 제거
        code = re.sub(r'/\*.*?\*/', '', cpp_code, flags=re.DOTALL)
        # 단일 라인 주석 제거
        code = re.sub(r'//.*', '', code)
        
        # 문자열 내용 제거 (" " 안의 내용과 ' ' 안의 내용)
        code = re.sub(r'"(.*?)"', '""', code)
        code = re.sub(r"'(.*?)'", "''", code)
        # 빈 줄 제거
        code = re.sub(r'\n\s*\n', '\n', code)
        # 불필요한 공백 및 탭 변환 (연속된 공백을 하나의 공백으로)
        code = re.sub(r'\s+', ' ', code)
        # 문자열 앞뒤 공백 제거
        cleaned_code = code.strip()
        
        return cleaned_code
    
    
TODAY = "2024_03_08"


with open(f'./bigdata/llama2/{TODAY}_b4bert_result_part1.txt', 'w') as file:
    pass


# tokenizer와 model은 미리 정의되어 있어야 합니다.
# device는 'cuda' 또는 'cpu'일 수 있습니다.

def predict(model, tokenizer, test_data):
    predictions = []
    
    
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
            
            # inputs1 = tokenizer(row['code1'], return_tensors='pt', max_length=512, padding='max_length', truncation=True).to(device)
            # inputs2 = tokenizer(row['code2'], return_tensors='pt', max_length=512, padding='max_length', truncation=True).to(device)
            inputtexts = f"""<s>[INST] <<SYS>>\\nYou are an intelligent programming assistant capable of understanding and analyzing C++ code. Your task is to determine if two given pieces of code solve the same problem. Respond with "Yes" if they solve the same problem and "No" otherwise.\\n<</SYS>>\\n\\nFirst code: \n{remove_comments(row['code1'])}Second code: \n{remove_comments(row['code2'])}\nDo these codes solve the same problem? Say yes or no.[/INST]"""            

            # 입력 텍스트의 토큰 수 계산
            input_tokens = tokenizer.encode(inputtexts, return_tensors="pt")
            input_length = len(input_tokens[0])

            # 원하는 출력 토큰 수
            desired_output_length = 5

            # 전체 max_length 설정 (입력 + 출력)
            total_max_length = input_length + desired_output_length

            # 파이프라인 설정 (예시에서는 나머지 매개변수를 유지)
            sequences = pipeline(
                inputtexts,
                do_sample=False,
                #top_k=1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_length=total_max_length,  # 수정된 부분
            )
            for seq in sequences:
                # 모델 출력에서 입력 텍스트 제거
                result_text = seq['generated_text'][len(inputtexts):]
                #result_text = seq['generated_text']
                predictions.append(result_text)
                with open(f'./bigdata/llama2/{TODAY}_b4bert_result_part1.txt', 'a') as file:
                    file.write(f'{index} : {result_text}\n')
        return predictions

    # 예제 사용
    test_data = pd.read_csv("./bigdata/test_part1.csv")
    # 모델과 tokenizer가 정의되어 있어야 합니다.
    predictions = predict(model, tokenizer, test_data)

    # 결과를 제출 파일로 저장
    submission = pd.read_csv('./bigdata/sample_submission_part1.csv')
    submission['similar'] = predictions
    submission.to_csv(f'./bigdata/llama2/{TODAY}_predictions_b4bert_submit_part1.csv', index=False)