import json
import sys
import os
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        all_features.append(cls_embedding.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)

def run_multidim_classification(data_path):
    texts = []
    all_labels = {dim: [] for dim in ["style", "format", "role", "sentiment", "length", "language"]}
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["generated_text"])
            for dim in all_labels:
                all_labels[dim].append(record[dim])
    
    print(f"数据集: {len(texts)} 条多维度联合标注文本")
    
    indices = list(range(len(texts)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train_texts = [texts[i] for i in train_idx]
    X_test_texts = [texts[i] for i in test_idx]
    
    print("提取特征...")
    X_train = extract_features(X_train_texts)
    X_test = extract_features(X_test_texts)
    print(f"训练特征: {X_train.shape}, 测试特征: {X_test.shape}")
    
    results = {}
    dim_names = {
        "style": "写作风格", "format": "格式要求", "role": "角色设定",
        "sentiment": "情感倾向", "length": "长度限制", "language": "语言特点"
    }
    
    for dim in all_labels:
        print(f"\n{'='*50}")
        print(f"训练维度: {dim_names[dim]} ({dim})")
        
        le = LabelEncoder()
        y_all = le.fit_transform(all_labels[dim])
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[dim] = {"accuracy": acc, "n_classes": len(le.classes_)}
        
        print(f"  类别数: {len(le.classes_)}")
        print(f"  类别: {list(le.classes_)}")
        print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n{'='*60}")
    print("联合解码统计")
    print(f"{'='*60}")
    
    joint_correct = 0
    joint_total = len(test_idx)
    
    for dim in all_labels:
        le = LabelEncoder()
        y_all = le.fit_transform(all_labels[dim])
        y_test = y_all[test_idx]
        
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        if dim == "style":
            all_preds = {dim: y_pred}
        else:
            all_preds[dim] = y_pred
    
    y_test_dict = {}
    for dim in all_labels:
        le = LabelEncoder()
        y_test_dict[dim] = le.fit_transform(all_labels[dim])[test_idx]
    
    for i in range(joint_total):
        correct = all(all_preds[dim][i] == y_test_dict[dim][i] for dim in all_labels)
        if correct:
            joint_correct += 1
    
    joint_acc = joint_correct / joint_total
    
    return results, joint_acc

if __name__ == "__main__":
    os.makedirs("data/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/results/multidim_report_{timestamp}.txt"
    
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
    print("多维度联合编码验证报告")
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results, joint_acc = run_multidim_classification("data/raw/multidim_6dim_data.jsonl")
    
    print(f"\n{'='*60}")
    print("六维度联合编码汇总")
    print(f"{'='*60}")
    print(f"{'维度':10s} {'类别数':>6s} {'准确率':>10s}")
    print("-"*30)
    dim_names = {
        "style": "写作风格", "format": "格式要求", "role": "角色设定",
        "sentiment": "情感倾向", "length": "长度限制", "language": "语言特点"
    }
    for dim, info in results.items():
        print(f"{dim_names[dim]:10s} {info['n_classes']:>6d} {info['accuracy']*100:>9.2f}%")
    
    print(f"\n联合解码准确率: {joint_acc:.4f} ({joint_acc*100:.2f}%)")
    print(f"(所有6个维度同时预测正确的比例)")
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    print(f"\n✅ 报告已保存至: {report_path}")