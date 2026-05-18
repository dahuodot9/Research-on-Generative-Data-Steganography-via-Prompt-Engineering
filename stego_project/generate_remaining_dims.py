import json
import os
import time
from datetime import datetime
from config import API_CONFIGS, OUTPUT_DIR, NUM_SAMPLES_PER_TEMPLATE, TEMPERATURE
from generate_data import call_wenxin

TEMPLATES = {
    "role": {
        "expert": [
            "请以该领域资深专家的身份回答，使用专业术语和权威表述：{question}",
            "假设你是一位有20年经验的行业专家，请展现专业深度：{question}",
            "以领域权威的口吻作答，引用专业知识体系：{question}",
        ],
        "student": [
            "请以一名正在学习的大学生身份回答，展现求知欲：{question}",
            "假设你是正在做课程作业的学生，用学生的视角回答：{question}",
            "以谦虚好学的学生口吻作答，可以带有思考和疑问：{question}",
        ],
        "journalist": [
            "请以新闻记者的身份回答，使用客观报道的语气和5W1H要素：{question}",
            "假设你正在撰写一篇深度报道，用记者视角回答：{question}",
            "以新闻导语体的方式作答，包含何时何地何人何事何因：{question}",
        ],
    },
    "sentiment": {
        "positive": [
            "请用积极乐观的态度回答，多谈优势和机遇：{question}",
            "以肯定和赞赏的语气作答，强调事物好的一面：{question}",
            "保持乐观向上的态度，重点讨论积极因素：{question}",
        ],
        "negative": [
            "请用批判和忧虑的态度回答，指出存在的问题和风险：{question}",
            "以审慎关切的语气作答，分析隐患和挑战：{question}",
            "保持批判性态度，重点讨论不足和需要改进之处：{question}",
        ],
        "neutral": [
            "请用客观中立的态度回答，不偏袒任何一方：{question}",
            "以冷静理性的语气作答，平衡呈现各方观点：{question}",
            "保持价值中立的立场，只陈述事实和数据：{question}",
        ],
    },
    "length": {
        "short": [
            "请用50-100字简短回答，直接说要点不要展开：{question}",
            "用一两句话精炼回答，控制在100字以内：{question}",
            "给出最简洁的回答，不必展开论述：{question}",
        ],
        "medium": [
            "请用200-300字完整回答，适当展开论述：{question}",
            "给出约两三百字的回答，包含必要解释：{question}",
            "回答控制在200-300字，既不过简也不冗长：{question}",
        ],
        "long": [
            "请用400-500字详细回答，全面展开论述：{question}",
            "给出一篇约四五百字的完整论述：{question}",
            "详细展开回答，篇幅在400-500字左右：{question}",
        ],
    },
    "language": {
        "concise": [
            "请用极为简洁直白的语言回答，每句话不超过15字：{question}",
            "直接说重点，用最精炼的词汇和短句作答：{question}",
            "用极简的表达方式，避免修饰和冗余：{question}",
        ],
        "detailed": [
            "请详细展开回答，给出充分的细节和例证：{question}",
            "用丰富的描述和事例来说明，力求全面深入：{question}",
            "充分展开论述，使用具体的例子支撑观点：{question}",
        ],
        "humorous": [
            "请用幽默风趣的语言回答，加入俏皮话和调侃：{question}",
            "以轻松诙谐的风格作答，让回答有趣又有料：{question}",
            "用调侃的方式表达观点，可以诙谐但不要冒犯：{question}",
        ],
    },
}

DIM_NAMES = {
    "role": "角色设定",
    "sentiment": "情感倾向", 
    "length": "长度限制",
    "language": "语言特点",
}

QUESTIONS = [
    "人工智能对未来教育的影响是什么？",
    "介绍一下中国的茶文化",
    "为什么要保护环境？",
    "如何看待远程办公的利弊",
    "介绍一下深度学习的基本原理",
    "健康饮食的重要性有哪些",
    "城市交通拥堵如何解决",
    "推荐一本好书并说明理由",
    "运动对身心健康的益处",
    "如何提高学习效率",
]

def generate_one_dim(dim_key):
    templates = TEMPLATES[dim_key]
    dim_name = DIM_NAMES[dim_key]
    api_config = API_CONFIGS["wenxin"]
    
    all_data = []
    total = sum(len(t) * len(QUESTIONS) * NUM_SAMPLES_PER_TEMPLATE for t in templates.values())
    
    print(f"\n{'='*50}")
    print(f"生成维度: {dim_name}")
    print(f"预计: {total} 条数据")
    print(f"{'='*50}")
    
    count = 0
    for style_name, template_list in templates.items():
        print(f"\n  [{style_name}] ...")
        for t_idx, template in enumerate(template_list):
            for question in QUESTIONS:
                prompt = template.format(question=question)
                for s_idx in range(NUM_SAMPLES_PER_TEMPLATE):
                    count += 1
                    print(f"  [{count}/{total}] {question[:15]}... ", end="")
                    
                    text = call_wenxin(prompt, api_config)
                    
                    if text is None:
                        print("❌")
                        continue
                    
                    print(f"✅ ({len(text)}字)")
                    
                    record = {
                        "model": "wenxin",
                        "dimension": dim_key,
                        "style": style_name,
                        "template_id": t_idx,
                        "template_text": template,
                        "question": question,
                        "full_prompt": prompt,
                        "sample_id": s_idx,
                        "generated_text": text,
                        "timestamp": datetime.now().isoformat(),
                    }
                    all_data.append(record)
                    time.sleep(0.3)
    
    output_path = os.path.join(OUTPUT_DIR, f"single_dim_{dim_key}_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\n✅ {dim_name} 完成！{len(all_data)}条 → {output_path}")
    return len(all_data)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0
    for dim in ["role", "sentiment", "length", "language"]:
        count = generate_one_dim(dim)
        total += count
    print(f"\n{'='*60}")
    print(f"全部完成！总计 {total} 条")
    print(f"{'='*60}")