import json
import sys
import os
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("加载数据...")
texts, labels, models = [], [], []

data_path = "data/raw/single_dim_style_data.jsonl"
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        texts.append(record["generated_text"])
        labels.append(record["style"])
        models.append(record["model"])

le = LabelEncoder()
y_all = le.fit_transform(labels)

total_by_model = {}
for m in models:
    total_by_model[m] = total_by_model.get(m, 0) + 1

print(f"总样本数: {len(texts)}")
print(f"类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
print(f"各模型样本数: {total_by_model}")

os.makedirs("data/results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"data/results/quick_classify_report_{timestamp}.txt"

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
print("单维度写作风格分类验证报告（TF-IDF + 逻辑回归）")
print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"特征方法: TF-IDF (max_features=5000)")
print(f"分类器: Logistic Regression (multi_class='multinomial')")
print("=" * 60)

results_summary = []

for model_name in ["wenxin", "doubao", "kimi"]:
    idx = [i for i, m in enumerate(models) if m == model_name]
    
    if len(idx) < 30:
        print(f"\n{'='*50}")
        print(f"⚠ {model_name}: 数据不足（仅{len(idx)}条），跳过")
        print(f"{'='*50}")
        continue
    
    X_sub = [texts[i] for i in idx]
    y_sub = y_all[idx]
    
    print(f"\n{'='*50}")
    print(f"🔬 模型: {model_name} ({len(X_sub)}条样本)")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
    )
    print(f"训练集: {len(X_train)}条, 测试集: {len(X_test)}条")

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("训练分类器...")
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 {model_name} 分类结果")
    print(f"{'-'*40}")
    print(f"准确率 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵:")
    print(f"              {'  '.join(le.classes_)}")
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:10s}  {str(row)}")
    
    results_summary.append({
        "model": model_name,
        "accuracy": f"{acc:.4f}",
        "train_size": len(X_train),
        "test_size": len(X_test),
    })

print(f"\n{'='*60}")
print("汇总：三个模型对比")
print(f"{'='*60}")
print(f"{'模型':12s} {'准确率':>8s} {'训练集':>8s} {'测试集':>8s}")
print("-" * 40)
for r in results_summary:
    print(f"{r['model']:12s} {r['accuracy']:>8s} {r['train_size']:>8s} {r['test_size']:>8s}")

if len(results_summary) >= 2:
    best = max(results_summary, key=lambda x: float(x['accuracy']))
    print(f"\n🏆 最佳模型: {best['model']} (准确率 {float(best['accuracy'])*100:.2f}%)")

print(f"\n{'='*60}")
print("分析要点:")
print(f"{'='*60}")
print("1. 哪个模型对 Prompt 风格控制最敏感？（准确率最高）")
print("2. 哪两种风格容易混淆？（看混淆矩阵非对角线值）")
print("3. 三个模型在风格区分上的一致性如何？")

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal

print(f"\n✅ 报告已保存至: {report_path}")
print(f"📋 各模型汇总:")
for r in results_summary:
    print(f"   {r['model']}: {float(r['accuracy'])*100:.1f}%")