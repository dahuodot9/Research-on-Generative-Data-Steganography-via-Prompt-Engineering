import json
import os
import time
import itertools
from datetime import datetime
from config import API_CONFIGS, OUTPUT_DIR, TEMPERATURE
from generate_data import call_wenxin

DIM_PROMPTS = {
    "style": {
        "formal": "请以严谨正式的风格回答",
        "oral": "用聊天唠嗑的语气回答",
    },
    "format": {
        "list": "用分点列表的形式，每条用数字编号",
        "paragraph": "用连贯的段落形式，不要使用列表符号",
    },
    "role": {
        "expert": "以该领域资深专家的身份，使用专业术语和权威性表述",
        "journalist": "以新闻记者的身份，使用客观报道的语气和5W1H要素",
    },
    "sentiment": {
        "positive": "持积极乐观的态度，多谈优势和机遇",
        "negative": "持批判和忧虑的态度，重点谈问题和风险",
    },
    "length": {
        "short": "回答请控制在50-100字以内，精炼简洁",
        "medium": "回答请控制在200-300字，适当展开",
    },
    "language": {
        "concise": "使用极为简洁直白的语言，每句话不超过15字",
        "humorous": "使用幽默风趣的语言，加入俏皮话和调侃",
    },
}

QUESTIONS = [
    "人工智能对未来教育的影响是什么？",
    "如何提高学习效率？",
]

def build_multidim_prompt(combo, question):
    """根据维度取值组合构建完整 Prompt"""
    parts = []
    for dim, value in combo.items():
        parts.append(DIM_PROMPTS[dim][value])
    
    instruction = "，".join(parts)
    return f"{instruction}：{question}"

def generate_multidim():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api_config = API_CONFIGS["wenxin"]
    
    selected = {
        "style": ["formal", "oral"],
        "format": ["list", "paragraph"],
        "role": ["expert", "journalist"],
        "sentiment": ["positive", "negative"],
        "length": ["short", "medium"],
        "language": ["concise", "humorous"],
    }
    
    dims = list(selected.keys())
    values_list = [selected[d] for d in dims]
    
    all_combinations = list(itertools.product(*values_list))
    total = len(all_combinations) * len(QUESTIONS)
    print(f"共 {len(all_combinations)} 种维度组合 × {len(QUESTIONS)} 个问题 = {total} 条数据")
    
    all_data = []
    count = 0
    
    for combo_tuple in all_combinations:
        combo = {dims[i]: combo_tuple[i] for i in range(len(dims))}
        
        for question in QUESTIONS:
            prompt = build_multidim_prompt(combo, question)
            count += 1
            
            print(f"[{count}/{total}] ", end="")
            
            text = call_wenxin(prompt, api_config)
            
            if text is None:
                print("❌")
                continue
            
            print(f"✅ ({len(text)}字)")
            
            record = {
                "model": "wenxin",
                "question": question,
                "full_prompt": prompt,
                "generated_text": text,
                "timestamp": datetime.now().isoformat(),
                "style": combo["style"],
                "format": combo["format"],
                "role": combo["role"],
                "sentiment": combo["sentiment"],
                "length": combo["length"],
                "language": combo["language"],
            }
            all_data.append(record)
            time.sleep(0.3)
    
    output_path = os.path.join(OUTPUT_DIR, "multidim_6dim_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 多维度联合数据集完成！{len(all_data)}条 → {output_path}")
    return output_path

if __name__ == "__main__":
    generate_multidim()