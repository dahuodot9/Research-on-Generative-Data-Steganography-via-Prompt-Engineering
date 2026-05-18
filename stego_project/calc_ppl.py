# calc_ppl.py
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"设备: {device}")

def calc_ppl(text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item()

data_files = {
    "写作风格": "data/raw/single_dim_style_data.jsonl",
    "格式要求": "data/raw/single_dim_format_data.jsonl",
    "角色设定": "data/raw/single_dim_role_data.jsonl",
    "情感倾向": "data/raw/single_dim_sentiment_data.jsonl",
    "长度限制": "data/raw/single_dim_length_data.jsonl",
    "语言特点": "data/raw/single_dim_language_data.jsonl",
}

results = []

for dim_name, file_path in data_files.items():
    if not os.path.exists(file_path):
        print(f"⚠ {dim_name}: 文件不存在，跳过")
        continue
    
    style_texts = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            style = record["style"]
            if style not in style_texts:
                style_texts[style] = []
            style_texts[style].append(record["generated_text"])
    
    print(f"\n{'='*50}")
    print(f"维度: {dim_name}")
    
    dim_ppls = []
    for style, texts in style_texts.items():
        # 每类随机取10条计算PPL，避免过慢
        sample_texts = np.random.choice(texts, min(10, len(texts)), replace=False)
        ppls = []
        for text in sample_texts:
            try:
                ppl = calc_ppl(text)
                ppls.append(ppl)
            except:
                continue
        
        avg_ppl = np.mean(ppls) if ppls else 0
        dim_ppls.extend(ppls)
        print(f"  {style:10s}: 平均PPL = {avg_ppl:.2f} ({len(ppls)}条)")
    
    overall_avg = np.mean(dim_ppls) if dim_ppls else 0
    results.append({"dimension": dim_name, "avg_ppl": overall_avg})
    print(f"  整体平均PPL: {overall_avg:.2f}")

print(f"\n{'='*60}")
print("PPL汇总")
print(f"{'='*60}")
print(f"{'维度':10s} {'平均PPL':>10s}")
print("-"*25)
for r in results:
    print(f"{r['dimension']:10s} {r['avg_ppl']:>10.2f}")

if results:
    overall = np.mean([r["avg_ppl"] for r in results])
    print(f"\n全部维度总平均PPL: {overall:.2f}")