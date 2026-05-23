# BERT IMDb Teacher

Paper C compares a Tsetlin Machine to a fine-tuned BERT-base-uncased
teacher on counterfactual IMDb. The BERT checkpoint is large (~4 GB),
reproducible, and lives outside the repo.

## Default location

```
~/model_archive/
`-- baseline_bert-base-uncased_imdb/seed_42/
    `-- checkpoints/checkpoint-<step>/
```

## Override

```bash
export BERT_MODEL_DIR=/path/to/teachers
python experiments/eval_paper_c_bert_on_counterfactual.py
```

## Reproducing the teacher

1. Load `bert-base-uncased` from HuggingFace.
2. Fine-tune on raw IMDb (HuggingFace `datasets.load_dataset("imdb")`).
3. Standard recipe: 3 epochs, lr=2e-5, batch=16, AdamW, max_len=512.
4. Save checkpoints under `seed_42/checkpoints/`.

A reproduction script is future work. Current results came from the
original run preserved at `~/model_archive/baseline_bert-base-uncased_imdb/`.
