import json
import sys
import os
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch

def extract_features(texts, model_name="bert-base-chinese", batch_size=32):
    print(f"加载模型 {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        all_features.append(cls_embedding.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"  已处理 {i+len(batch_texts)}/{len(texts)}")
    
    return np.concatenate(all_features, axis=0)

def load_data(data_path):
    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["generated_text"])
            labels.append(record["style"])
    return texts, labels

def run_classification(data_path):
    print(f"加载数据: {data_path}")
    all_texts, all_labels = load_data(data_path)
    print(f"总样本: {len(all_texts)}")
    
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    print(f"类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        all_texts, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n提取训练集特征...")
    X_train = extract_features(X_train_texts)
    print(f"训练特征: {X_train.shape}")
    
    print("\n提取测试集特征...")
    X_test = extract_features(X_test_texts)
    print(f"测试特征: {X_test.shape}")
    
    print("\n训练分类器...")
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"📊 格式要求维度分类结果")
    print(f"{'='*50}")
    print(f"准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(f"         {'  '.join(le.classes_)}")
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:8s}  {str(row)}")
    
    return acc, le, cm

if __name__ == "__main__":
    os.makedirs("data/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/results/format_classify_report_{timestamp}.txt"
    
    class Tee:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Tee(report_path)
    
    print("=" * 60)
    print("格式要求维度可区分性验证报告")
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    acc, le, cm = run_classification("data/raw/single_dim_format_data.jsonl")
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    
    print(f"\n✅ 报告已保存至: {report_path}")
    print(f"📊 格式维度分类准确率: {acc*100:.2f}%")