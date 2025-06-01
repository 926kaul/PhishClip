import clip
import torch
import json
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

with open("test_samples_with_domain.json", "r") as f:
    test_samples = json.load(f)

label_map = {"benign": 0, "phishing": 1}
labels = list(label_map.keys())

results = []
correct = 0
total = 0
total_samples = len(test_samples)
step = max(1, total_samples // 100)

for idx, sample in enumerate(test_samples):
    img_path = sample["img_path"]
    domain = sample.get("domain", "unknown")
    gt_label = sample["label"]

    prompts = [f"{label} site of {domain}" for label in labels]

    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = (image_features @ text_features.T).softmax(dim=-1)
    pred_idx = logits.argmax().item()
    pred_label = labels[pred_idx]
    confidence = logits[0, pred_idx].item()

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

    # 1%마다 저장
    if (idx + 1) % step == 0 or idx + 1 == total_samples:
        print(f"🔵 [{int((idx+1)/total_samples*100)}%] {idx+1}/{total_samples} processed, "
              f"Current accuracy: {correct/total:.4f}, "
              f"Latest GT: {gt_label}, Pred: {pred_label}, Conf: {confidence:.4f}")
        with open('clip_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nZero-shot CLIP accuracy: {correct}/{total} = {correct/total:.4f}")
