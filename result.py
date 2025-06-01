import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# 파일명 매핑 (필요에 따라 수정)
result_files = {
    "clip": "clip_test_results.json",
    "coop_1": "coop_test_results_1.json",
    "coop_2": "coop_test_results_2.json",
    "coop_3": "coop_test_results_3.json",
    "coop_4": "coop_test_results_4.json",
    "coop_5": "coop_test_results_5.json",
    "phishpedia": "phishpedia_test_results.json"
}

# Confusion Matrix, Accuracy, Confidence 평균값 저장
metrics = {}

def load_preds_phishpedia(file):
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
    y_true = [d["label"] for d in data]
    y_pred = [d["pred_category"] for d in data]
    confs = [float(d["confidence"]) for d in data if d.get("confidence") is not None]
    return y_true, y_pred, confs

def load_preds_coop_clip(file):
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
    y_true = [d.get("ground_truth", d.get("label")) for d in data]
    y_pred = [d.get("prediction") for d in data]
    confs = [float(c) for c in [d.get("confidence") for d in data if d.get("confidence") is not None]]
    return y_true, y_pred, confs

# 클래스 레이블 순서
labels = ["benign", "phishing"]

# 파일별로 처리
for name, path in result_files.items():
    if name == "phishpedia":
        y_true, y_pred, confs = load_preds_phishpedia(path)
    else:
        y_true, y_pred, confs = load_preds_coop_clip(path)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    avg_conf = np.mean(confs)
    metrics[name] = {"cm": cm, "acc": acc, "conf": avg_conf}

# 이름, 값 리스트 추출
names = list(metrics.keys())
accs = [metrics[n]["acc"] for n in names]
confs = [metrics[n]["conf"] for n in names]

# 1. Accuracy Bar Plot
plt.figure(figsize=(8,5))
bars = plt.bar(names, accs, color='skyblue')
plt.title("Accuracy by Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
for bar, val in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("accuracy_by_model.png")
plt.close()

# 2. Confidence Bar Plot
plt.figure(figsize=(8,5))
bars = plt.bar(names, confs, color='salmon')
plt.title("Mean Confidence by Model")
plt.ylabel("Mean Confidence")
plt.ylim(0, 1.05)
for bar, val in zip(bars, confs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("confidence_by_model.png")
plt.close()

# 3. Confusion Matrix (각 모델별 PNG로 저장)
for name, stat in metrics.items():
    plt.figure(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(stat["cm"], display_labels=labels)
    disp.plot(ax=plt.gca(), colorbar=False)
    plt.title(f"{name}\nAcc={stat['acc']:.3f}, Conf={stat['conf']:.3f}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()
