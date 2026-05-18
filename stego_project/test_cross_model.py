from config import API_CONFIGS
from generate_data import call_wenxin, call_doubao, call_kimi

prompts = {
    "写作风格-正式": "请以严谨正式的风格回答，使用规范的书面用语：人工智能对未来教育的影响是什么？",
    "写作风格-口语": "用聊天唠嗑的语气回答，就像跟朋友聊天一样：人工智能对未来教育的影响是什么？",
    "格式要求-列表": "请以分点列表的形式回答，每条用数字编号：如何提高学习效率？",
}

for prompt_name, prompt_text in prompts.items():
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt_name}")
    print(f"内容: {prompt_text[:50]}...")
    print(f"{'='*60}")
    
    for model_name, caller in [("wenxin", call_wenxin), ("doubao", call_doubao), ("kimi", call_kimi)]:
        print(f"\n--- {model_name} ---")
        try:
            result = caller(prompt_text, API_CONFIGS[model_name])
            if result:
                print(result[:200])
            else:
                print("调用失败")
        except Exception as e:
            print(f"错误: {e}")
    
    print()