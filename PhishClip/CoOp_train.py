import torch
import clip
from torch import nn, optim
from PIL import Image
import json
import time
import random

# 1. dataset load
with open("train_samples_with_domain.json") as f:
    samples = json.load(f)

subset_size = int(len(samples) * 1.0)
samples_subset = random.sample(samples, subset_size)

label_map = {"benign": 0, "phishing": 1}
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
clip_model.eval()
num_classes = len(label_map)
prompt_len = 8
embed_dim = clip_model.ln_final.weight.shape[0]
soft_prompts = nn.Parameter(torch.randn(num_classes, prompt_len, embed_dim, device="cuda"))
optimizer = optim.Adam([soft_prompts], lr=1e-3)

def make_prompt(label, domain):
    return f"{label} site of {domain}"

total_samples = len(samples_subset)
step = max(1, total_samples // 100)

for epoch in range(5):
    # shuffle data for each epoch
    random.shuffle(samples_subset)
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for idx, sample in enumerate(samples_subset):
        try:
            img_path = sample["img_path"]
            label = sample["label"]
            domain = sample.get("domain", "unknown")
            label_idx = label_map[label]
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
            with torch.no_grad():
                img_embed = clip_model.encode_image(img)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            prompts = [make_prompt(lbl, domain) for lbl in label_map.keys()]
            tokens = clip.tokenize(prompts).cuda()
            with torch.no_grad():
                text_feats = clip_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            for i in range(num_classes):
                text_feats[i] = text_feats[i] + soft_prompts[i].mean(dim=0)
            logits = (img_embed @ text_feats.T)
            target = torch.tensor([label_idx], device="cuda")
            loss = nn.CrossEntropyLoss()(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_idx = logits.softmax(dim=-1).argmax().item()
            if pred_idx == label_idx:
                correct += 1
            total += 1

            # report result for every 1% step
            if (idx + 1) % step == 0 or idx + 1 == total_samples:
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch+1}] [{int((idx+1)/total_samples*100)}%] "
                    f"{idx+1}/{total_samples} - "
                    f"Curr Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f} | "
                    f"GT: {label}, Pred: {list(label_map.keys())[pred_idx]} | "
                    f"Elapsed: {elapsed:.1f} sec")
        except:
            continue

    print(f"\nEpoch {epoch+1} done | Avg Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}\n")
    torch.save(soft_prompts, f"soft_prompt_tensor_full{epoch+1}.pth")
    print("✅ soft prompt saved: soft_prompt_tensor.pth")
