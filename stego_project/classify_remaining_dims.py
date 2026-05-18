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

def extract_features(texts, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        all_features.append(cls_embedding.cpu().numpy())
    return np.concatenate(all_features, axis=0)

def classify_one_dim(data_path, dim_name):
    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["generated_text"])
            labels.append(record["style"])
    
    if len(texts) == 0:
        return None
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  样本: {len(texts)}条, 训练: {len(X_train_texts)}, 测试: {len(X_test_texts)}")
    print("  提取特征...")
    X_train = extract_features(X_train_texts)
    X_test = extract_features(X_test_texts)
    
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 {dim_name}")
    print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  类别: {list(le.classes_)}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"  混淆矩阵:")
    print(f"         {'  '.join(le.classes_)}")
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:8s} {str(row)}")
    
    return {"dimension": dim_name, "accuracy": acc, "classes": le.classes_, "n_samples": len(texts)}

if __name__ == "__main__":
    os.makedirs("data/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/results/remaining_dims_{timestamp}.txt"
    
    class Tee:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, "w", encoding="utf-8")
        def write(self, msg):
            self.terminal.write(msg)
            self.log.write(msg)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Tee(report_path)
    
    print("=" * 60)
    print("剩余四维度单维度分类报告")
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    dims = [
        ("data/raw/single_dim_role_data.jsonl", "角色设定"),
        ("data/raw/single_dim_sentiment_data.jsonl", "情感倾向"),
        ("data/raw/single_dim_length_data.jsonl", "长度限制"),
        ("data/raw/single_dim_language_data.jsonl", "语言特点"),
    ]
    
    results = []
    for path, name in dims:
        if os.path.exists(path):
            print(f"\n{'='*50}")
            print(f"维度: {name}")
            r = classify_one_dim(path, name)
            if r:
                results.append(r)
        else:
            print(f"\n⚠ {name}: 数据文件不存在 ({path})")
    
    print(f"\n\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    print(f"{'维度':10s} {'样本数':>6s} {'准确率':>10s}")
    print("-"*30)
    for r in results:
        print(f"{r['dimension']:10s} {r['n_samples']:>6d} {r['accuracy']*100:>9.2f}%")
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✅ 报告已保存至: {report_path}")