import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
import os

def extract_features(texts, model_name="bert-base-chinese", batch_size=32):
    """
    用预训练BERT提取文本的[CLS]向量作为特征。
    返回: numpy array, shape (n_texts, 768)
    """
    print(f"加载模型 {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_features = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        all_features.append(cls_embedding.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"  已处理 {i+len(batch_texts)}/{len(texts)}")
    
    return np.concatenate(all_features, axis=0)

def load_data(data_path):
    """加载生成的JSONL数据"""
    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["generated_text"])
            labels.append(record["style"])  # formal / oral / academic
    return texts, labels


def run_classification_experiment(data_path, model_name_filter=None):
    """
    主实验函数。
    model_name_filter: 如果指定(如"doubao")，则只用该模型数据进行训练和测试。
    """
    print(f"加载数据: {data_path}")
    all_texts, all_labels = load_data(data_path)
    
    if model_name_filter:
        filtered_texts, filtered_labels = [], []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["model"] == model_name_filter:
                    filtered_texts.append(record["generated_text"])
                    filtered_labels.append(record["style"])
        all_texts, all_labels = filtered_texts, filtered_labels
        print(f"已过滤，仅使用 {model_name_filter} 的数据: {len(all_texts)} 条")
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    print(f"类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        all_texts, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n提取训练集特征...")
    X_train = extract_features(X_train_texts)
    print(f"训练特征shape: {X_train.shape}")
    
    print("\n提取测试集特征...")
    X_test = extract_features(X_test_texts)
    print(f"测试特征shape: {X_test.shape}")

    print("\n训练Logistic Regression分类器...")
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"📊 分类结果 (模型: {model_name_filter or '全部'})")
    print(f"{'='*50}")
    print(f"准确率 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n详细报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"       {'  '.join(le.classes_)}")
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:8s} {row}")
    
    return clf, le, acc


if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    os.makedirs("data/results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/results/classification_report_{timestamp}.txt"
    
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
    print("单维度写作风格分类验证报告")
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"特征模型: bert-base-chinese")
    print(f"分类器: Logistic Regression")
    print("=" * 60)

    print("\n" + "🔬 实验：文心(wenxin)模型内部单维度分类")
    print("-" * 40)
    run_classification_experiment(
        data_path="data/raw/single_dim_style_data.jsonl",
        model_name_filter="wenxin"
    )

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    
    print(f"\n✅ 报告已保存至: {report_path}")