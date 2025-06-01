import torch
import clip
from PIL import Image
import json
import time

# 1. 데이터 로드
with open("test_samples_with_domain.json") as f:
    test_samples = json.load(f)

label_map = {"benign": 0, "phishing": 1}
labels = list(label_map.keys())

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

num_classes = len(label_map)
prompt_len = 8
embed_dim = clip_model.ln_final.weight.shape[0]

# 2. soft prompt 파라미터 불러오기
soft_prompts = torch.load("soft_prompt_tensor.pth").to(device)  # (num_classes, prompt_len, embed_dim)

def make_prompt(label, domain):
    return f"{label} site of {domain}"

results = []
correct = 0
total = 0
total_samples = len(test_samples)
step = max(1, total_samples // 100)
start_time = time.time()

for idx, sample in enumerate(test_samples):
    img_path = sample["img_path"]
    domain = sample.get("domain", "unknown")
    gt_label = sample["label"]

    prompts = [make_prompt(lbl, domain) for lbl in labels]
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embed = clip_model.encode_image(image)
    img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    # 각 클래스별 soft prompt를 텍스트 임베딩에 덧셈
    for i in range(num_classes):
        text_feats[i] = text_feats[i] + soft_prompts[i].mean(dim=0)
    logits = (img_embed @ text_feats.T)
    probs = logits.softmax(dim=-1).squeeze()
    pred_idx = probs.argmax().item()
    pred_label = labels[pred_idx]
    confidence = probs[pred_idx].item()

    results.append({
        "img_path": img_path,
        "domain": domain,
        "prediction": pred_label,
        "confidence": float(confidence),
        "ground_truth": gt_label,
    })

    if pred_label == gt_label:
        correct += 1
    total += 1

    # 1%마다 진행상황 및 JSON 저장
    if (idx + 1) % step == 0 or idx + 1 == total_samples:
        elapsed = time.time() - start_time
        print(f"[{int((idx+1)/total_samples*100)}%] {idx+1}/{total_samples} - "
              f"Acc: {correct/total:.4f} | GT: {gt_label}, Pred: {pred_label}, Conf: {confidence:.4f} | Elapsed: {elapsed:.1f}s")
        with open("coop_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n최종 Zero-shot CoOp Acc: {correct}/{total} = {correct/total:.4f}")
