import os
import json
import ast
from urllib.parse import urlparse

# 실제 파싱 및 도메인 추가
with open("train_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

def extract_domain_from_url(url):
    try:
        host = urlparse(url).hostname
        return host if host else url
    except Exception:
        return url

def get_domain_from_sample(sample):
    img_path = sample["img_path"]
    label = sample["label"]
    folder = os.path.dirname(img_path)
    if label == "benign":
        # 폴더명이 곧 도메인
        return os.path.basename(folder)
    elif label == "phishing":
        # info.txt의 url 필드에서 도메인 추출
        info_path = os.path.join(folder, "info.txt")
        if os.path.exists(info_path):
            try:
                with open(info_path, encoding="utf-8") as f:
                    content = f.read().strip()
                if content.startswith("{"):
                    info = ast.literal_eval(content)
                    url = info.get("url", "")
                else:
                    url = content
                if url:
                    return extract_domain_from_url(url)
            except Exception as e:
                print(f"⚠️ info.txt 파싱 에러: {info_path}\n{e}")
        # 실패 시
        return "unknown"
    else:
        return "unknown"

for sample in samples:
    sample["domain"] = get_domain_from_sample(sample)

with open("train_samples_with_domain.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print("✅ 도메인 필드 추가 완료! 예시:", samples[:3])
