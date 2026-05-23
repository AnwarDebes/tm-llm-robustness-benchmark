#!/usr/bin/env python3
"""Download and cache all robustness evaluation datasets for Paper C.

Datasets:
- AdvGLUE SST-2 (adversarial sentiment, 148 samples)
- Counterfactual IMDb (Kaushik et al. 2020, 488 test samples)
- Standard IMDb (25K train, 25K test)
- Standard SST-2 (via GLUE)
- AG News (already available locally)
"""

import os
import json
from datasets import load_dataset

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "robustness"
)


def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. AdvGLUE SST-2
    print("Downloading AdvGLUE SST-2...")
    ds = load_dataset("adv_glue", "adv_sst2")
    adv_sst2 = []
    for ex in ds["validation"]:
        adv_sst2.append({"text": ex["sentence"], "label": ex["label"]})
    with open(os.path.join(DATA_DIR, "advglue_sst2.json"), "w") as f:
        json.dump(adv_sst2, f, indent=2)
    print(f"  Saved {len(adv_sst2)} adversarial SST-2 examples")

    # 2. Counterfactual IMDb
    print("Downloading Counterfactual IMDb...")
    ds = load_dataset("tasksource/counterfactually-augmented-imdb")
    cf_data = {}
    for split in ds:
        cf_data[split] = []
        for ex in ds[split]:
            cf_data[split].append({
                "text": ex["Text"],
                "label": ex["Sentiment"],
            })
    with open(os.path.join(DATA_DIR, "counterfactual_imdb.json"), "w") as f:
        json.dump(cf_data, f, indent=2)
    print(f"  Saved train={len(cf_data.get('train',[]))}, "
          f"val={len(cf_data.get('validation',[]))}, "
          f"test={len(cf_data.get('test',[]))}")

    # 3. Standard IMDb
    print("Downloading IMDb...")
    ds = load_dataset("imdb")
    imdb = {
        "train": [{"text": ex["text"], "label": ex["label"]} for ex in ds["train"]],
        "test": [{"text": ex["text"], "label": ex["label"]} for ex in ds["test"]],
    }
    with open(os.path.join(DATA_DIR, "imdb.json"), "w") as f:
        json.dump({"train_size": len(imdb["train"]), "test_size": len(imdb["test"])}, f)
    print(f"  IMDb train={len(imdb['train'])}, test={len(imdb['test'])}")
    # Don't save full IMDb to JSON (too large); use HuggingFace cache

    # 4. SST-2 (from GLUE)
    print("Downloading SST-2...")
    ds = load_dataset("glue", "sst2")
    sst2 = {
        "train_size": len(ds["train"]),
        "validation_size": len(ds["validation"]),
    }
    with open(os.path.join(DATA_DIR, "sst2_info.json"), "w") as f:
        json.dump(sst2, f, indent=2)
    print(f"  SST-2 train={sst2['train_size']}, val={sst2['validation_size']}")

    print(f"\nAll robustness data saved to {DATA_DIR}")


if __name__ == "__main__":
    download_all()
