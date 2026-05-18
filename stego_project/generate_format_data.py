import json
import os
import time
from datetime import datetime
from config import API_CONFIGS, OUTPUT_DIR, NUM_SAMPLES_PER_TEMPLATE, TEMPERATURE
from prompt_templates import FORMAT_TEMPLATES, DEFAULT_QUESTIONS
from generate_data import call_wenxin

def generate_format():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open("data/questions.txt", "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    all_data = []
    api_config = API_CONFIGS["wenxin"]
    
    print("=" * 50)
    print("开始生成格式要求维度数据")
    print("模型: wenxin")
    print("=" * 50)
    
    for format_name, templates in FORMAT_TEMPLATES.items():
        print(f"\n  [{format_name}] 格式...")
        
        for template_idx, template in enumerate(templates):
            print(f"    模板{template_idx+1}/{len(templates)}: {template[:50]}...")
            
            for question_idx, question in enumerate(questions):
                prompt = template.format(question=question)
                
                for sample_idx in range(NUM_SAMPLES_PER_TEMPLATE):
                    print(f"      [{question_idx+1}/{len(questions)}] {question[:20]}... 样本{sample_idx+1}", end="")
                    
                    generated_text = call_wenxin(prompt, api_config)
                    
                    if generated_text is None:
                        print(" ❌ 失败")
                        continue
                    
                    print(f" ✅ ({len(generated_text)}字)")
                    
                    record = {
                        "model": "wenxin",
                        "dimension": "format",
                        "style": format_name,
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
        
        print(f"  [{format_name}] 完成，当前总样本数: {len(all_data)}")
    
    output_path = os.path.join(OUTPUT_DIR, "single_dim_format_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 全部完成！共生成 {len(all_data)} 条数据")
    print(f"保存至: {output_path}")
    
    from collections import Counter
    style_counts = Counter(r["style"] for r in all_data)
    print(f"各格式样本数: {dict(style_counts)}")

if __name__ == "__main__":
    generate_format()