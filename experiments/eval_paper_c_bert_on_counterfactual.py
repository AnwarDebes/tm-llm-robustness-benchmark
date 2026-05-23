#!/usr/bin/env python3
"""Paper C: Evaluate fine-tuned BERT on counterfactual IMDb.

Loads the fine-tuned BERT checkpoint and evaluates on:
1. Clean IMDb test set (verification)
2. Counterfactual IMDb test set (488 samples)
"""

import os, sys, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# BERT teacher checkpoints were moved out of the repo to keep it lightweight.
# Override BERT_MODEL_DIR if you store them elsewhere. See MODELS.md.
BERT_MODEL_DIR = os.environ.get("BERT_MODEL_DIR", "/home/anward/model_archive")

# Find best checkpoint
ckpt_dir = os.path.join(BERT_MODEL_DIR,
                        "baseline_bert-base-uncased_imdb", "seed_42", "checkpoints")
ckpts = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("-")[1]))
best_ckpt = os.path.join(ckpt_dir, ckpts[-1])  # last checkpoint (best model)
print(f"Loading checkpoint: {best_ckpt}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(best_ckpt).cuda().eval()

# 1. Verify on clean IMDb
print("\n1. Clean IMDb test set...")
ds = load_dataset("imdb")
test_texts = ds["test"]["text"][:1000]  # Subset for speed
test_labels = ds["test"]["label"][:1000]

correct = 0
with torch.no_grad():
    for i in range(0, len(test_texts), 32):
        batch = test_texts[i:i+32]
        labels = test_labels[i:i+32]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                          max_length=256, padding=True).to("cuda")
        logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        correct += (preds == np.array(labels)).sum()

clean_acc = correct / len(test_texts)
print(f"   Clean accuracy (1000 samples): {clean_acc*100:.2f}%")

# 2. Counterfactual IMDb
print("\n2. Counterfactual IMDb...")
cf_path = os.path.join(PROJECT_ROOT, "data", "robustness", "counterfactual_imdb.json")
with open(cf_path) as f:
    cf_data = json.load(f)

cf_test = cf_data.get("test", [])
cf_texts = [ex["text"] for ex in cf_test]
cf_labels_raw = [ex["label"] for ex in cf_test]
label_map = {"Negative": 0, "Positive": 1, 0: 0, 1: 1}
cf_labels = [label_map.get(l, l) for l in cf_labels_raw]

correct = 0
with torch.no_grad():
    for i in range(0, len(cf_texts), 32):
        batch = cf_texts[i:i+32]
        labels = cf_labels[i:i+32]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                          max_length=256, padding=True).to("cuda")
        logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        correct += (preds == np.array(labels)).sum()

cf_acc = correct / len(cf_texts)
print(f"   Counterfactual accuracy ({len(cf_texts)} samples): {cf_acc*100:.2f}%")

robustness_ratio = cf_acc / max(clean_acc, 1e-10)

print(f"\n=== BERT IMDb Robustness ===")
print(f"  Clean: {clean_acc*100:.2f}%")
print(f"  Counterfactual: {cf_acc*100:.2f}%")
print(f"  Robustness ratio: {robustness_ratio:.4f}")

results = {
    "model": "BERT-base",
    "dataset": "IMDb",
    "clean_accuracy": float(clean_acc),
    "counterfactual_accuracy": float(cf_acc),
    "robustness_ratio": float(robustness_ratio),
}

out_path = os.path.join(PROJECT_ROOT, "experiments", "paper_c_bert_imdb", "seed_42", "summary.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
