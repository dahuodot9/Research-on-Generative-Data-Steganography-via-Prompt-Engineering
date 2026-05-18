import json
import os
import time
from datetime import datetime
from config import (
    API_CONFIGS, MODELS_TO_TEST, OUTPUT_DIR, 
    NUM_SAMPLES_PER_TEMPLATE, TEMPERATURE
)
from prompt_templates import STYLE_TEMPLATES, DEFAULT_QUESTIONS

def call_doubao(prompt, api_config):
    """豆包API调用 (火山引擎方舟平台)"""
    import requests
    
    url = f"{api_config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": api_config["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 500,
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"豆包API错误: {response.status_code} {response.text}")
        return None


def call_wenxin(prompt, api_config):
    """
    文心一言 API 调用 (百度千帆 OpenAI 兼容接口)
    支持 bce-v3 格式密钥，适用于 ERNIE-Lite-Pro-128K
    """
    from openai import OpenAI
    
    client = OpenAI(
        api_key=api_config["api_key"],                           
        base_url="https://qianfan.baidubce.com/v2",             
    )
    
    try:
        response = client.chat.completions.create(
            model="ernie-lite-pro-128k",                          
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"文心API错误: {e}")
        return None


def call_kimi(prompt, api_config):
    """Kimi API调用 (兼容OpenAI格式)"""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"]
    )
    
    try:
        response = client.chat.completions.create(
            model=api_config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Kimi API错误: {e}")
        return None


API_CALLERS = {
    "doubao": call_doubao,
    "wenxin": call_wenxin,
    "kimi": call_kimi,
}


def generate_all():
    """主函数：遍历模型-风格-模板-问题 生成文本"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open("data/questions.txt", "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    all_data = []
    
    for model_name in MODELS_TO_TEST:
        print(f"\n{'='*50}")
        print(f"开始测试模型: {model_name}")
        print(f"{'='*50}")
        
        caller = API_CALLERS.get(model_name)
        if caller is None:
            print(f"模型 {model_name} 的调用函数未定义，跳过")
            continue
        
        api_config = API_CONFIGS[model_name]
        
        for style_name, templates in STYLE_TEMPLATES.items():
            print(f"\n  [{style_name}] 风格...")
            
            for template_idx, template in enumerate(templates):
                print(f"    模板{template_idx+1}/{len(templates)}: {template[:40]}...")
                
                for question_idx, question in enumerate(questions):
                    prompt = template.format(question=question)
                    
                    for sample_idx in range(NUM_SAMPLES_PER_TEMPLATE):
                        print(f"      [{question_idx+1}/{len(questions)}] {question[:20]}... 样本{sample_idx+1}", end="")
                        
                        generated_text = caller(prompt, api_config)
                        
                        if generated_text is None:
                            print(" ❌ 失败")
                            continue
                        
                        print(f" ✅ ({len(generated_text)}字)")
                        
                        record = {
                            "model": model_name,
                            "style": style_name,
                            "template_id": template_idx,
                            "template_text": template,
                            "question": question,
                            "full_prompt": prompt,
                            "sample_id": sample_idx,
                            "generated_text": generated_text,
                            "timestamp": datetime.now().isoformat(),
                        }
                        all_data.append(record)
                        
                        time.sleep(0.3)  
            
            print(f"  [{style_name}] 完成，当前总样本数: {len(all_data)}")
    
    if len(all_data) == 0:
        print("\n❌ 未生成任何数据！请检查API配置")
        return
    
    output_path = os.path.join(OUTPUT_DIR, "single_dim_style_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 全部完成！共生成 {len(all_data)} 条数据")
    print(f"保存至: {output_path}")
    
    from collections import Counter
    style_counts = Counter(r["style"] for r in all_data)
    model_counts = Counter(r["model"] for r in all_data)
    print(f"\n各风格样本数: {dict(style_counts)}")
    print(f"各模型样本数: {dict(model_counts)}")


if __name__ == "__main__":
    generate_all()