# Research-on-Generative-Data-Steganography-via-Prompt-Engineering

本项目是基于大语言模型API的生成式文本隐写系统，通过控制Prompt的六个维度组合实现秘密信息的隐蔽嵌入与零样本提取。系统支持文心一言、豆包、Kimi等多模型API调用、批量数据生成与标注、BERT特征提取与风格分类、多维度联合编码验证、困惑度评估及跨模型响应对比。可用于隐蔽通信、信息安全、隐私保护等场景的学术研究与原型验证。

## 项目目录结构
```text
stego_project/
├── data/
│   ├── questions.txt
│   └── raw/
│       ├── multidim_6dim_data.jsonl
│       ├── single_dim_format_data.jsonl
│       ├── single_dim_language_data.jsonl
│       ├── single_dim_length_data.jsonl
│       ├── single_dim_role_data.jsonl
│       ├── single_dim_sentiment_data.jsonl
│       └── single_dim_style_data.jsonl
│
├── results/
│   ├── classification_report_20260506_013002.txt
│   ├── format_classify_report_20260506_022918.txt
│   ├── multidim_report_20260506_033246.txt
│   └── remaining_dims_20260506_053728.txt
│
├── calc_ppl.py
├── classify_remaining_dims.py
├── config.py
├── generate_data.py
├── generate_format_data.py
├── generate_multidim_data.py
├── generate_remaining_dims.py
├── prompt_templates.py
├── quick_classify.py
├── test_api.py
├── test_cross_model.py
├── train_classifier.py
├── train_format_classifier.py
└── train_multidim_classifier.py
```

# 项目声明
- **项目名称**：基于提示工程的生成式数据隐写研究
- **项目作者**：Yang Shijie
- **作者单位**：暨南大学网络空间安全学院
- **开发语言**：Python
- **框架**：PyTorch、HuggingFace Transformers、Scikit-learn
- **核心技术**：生成式文本隐写、提示工程、多维度Prompt编码、零样本提取、对比学习、BERT特征提取
