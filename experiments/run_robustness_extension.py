#!/usr/bin/env python3
"""Run a TextAttack recipe against a Tsetlin-family student.

This script wires up three GraphTM-based students (Paper A, Paper B,
decoder Qwen) to the paper-c attack harness. It loads the BertGCN
canonical R8 / R52 split (or IMDb), wraps the saved student via the
``robustness.students`` package, and dispatches to ``attack_runner``.

Usage:

    python experiments/run_robustness_extension.py \\
        --student subword_dep \\
        --recipe textfooler \\
        --dataset R8 \\
        --checkpoint /path/to/saved_student

A complete attack run requires a saved Tsetlin student and a GPU. This
script does not invent results; it logs the path it would have written
when a checkpoint is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from eval.logger import ExperimentLogger  # noqa: E402
from utils.load_bertgcn_splits import load_split  # noqa: E402


STUDENTS = {
    "subword_dep": "Paper A subword + typed-dependency GraphTM",
    "bert_attention": "Paper B BERT-attention GraphTM",
    "qwen_attention": "Decoder-distill Qwen-attention GraphTM",
}

RECIPES = ["textfooler", "bert_attack"]

DATASETS = ["R8", "R52", "imdb"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a TextAttack recipe against a Tsetlin GraphTM student."
    )
    parser.add_argument(
        "--student", required=True, choices=list(STUDENTS.keys()),
        help="Which student family to load.",
    )
    parser.add_argument(
        "--recipe", required=True, choices=RECIPES,
        help="TextAttack recipe to run.",
    )
    parser.add_argument(
        "--dataset", required=True, choices=DATASETS,
        help="Dataset split. R8 / R52 use BertGCN canonical splits; "
             "imdb uses the bundled IMDb test set.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Directory holding tm_state.pkl, vocab.json, labels.json, "
             "config.json for the chosen student.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=200,
        help="Number of correctly-classified test samples to attack.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Where to write the run log. Defaults to experiments/"
             "robustness_extension/<student>_<dataset>_<recipe>/seed_<seed>/",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Verify wiring (dataset load + student instantiation) and "
             "exit before launching TextAttack.",
    )
    return parser.parse_args()


def load_dataset_pairs(dataset_name) -> Tuple[List[str], List[int], int, List[str]]:
    """Load (texts, labels, n_classes, label_names) for the requested dataset."""
    name = dataset_name.lower()
    if name in ("r8", "r52"):
        split = load_split(dataset_name.upper() if name != "r52" else "R52")
        return (
            list(split["test_texts"]),
            list(split["test_labels"]),
            int(split["n_classes"]),
            list(split["label_names"]),
        )
    if name == "imdb":
        # IMDb test set is bundled in data/robustness/imdb.json.
        path = os.path.join(PROJECT_ROOT, "data", "robustness", "imdb.json")
        with open(path) as f:
            data = json.load(f)
        test = data.get("test", [])
        if not test:
            raise ValueError(
                f"{path} has no 'test' split. Check the dataset bundle."
            )
        texts = [ex["text"] for ex in test]
        labels = [int(ex["label"]) for ex in test]
        return texts, labels, len(set(labels)), sorted(set(str(l) for l in labels))
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "experiments", "robustness_extension",
        f"{args.student}_{args.dataset.lower()}_{args.recipe}",
        f"seed_{args.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(os.path.join(out_dir, "log.jsonl"))
    logger.log_config({
        "student": args.student,
        "student_description": STUDENTS[args.student],
        "recipe": args.recipe,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "seed": args.seed,
    })

    print(f"=== Extension run: {STUDENTS[args.student]} ===")
    print(f"  dataset:    {args.dataset}")
    print(f"  recipe:     {args.recipe}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  seed:       {args.seed}")

    print("\nLoading dataset...")
    texts, labels, n_classes, label_names = load_dataset_pairs(args.dataset)
    print(f"  n={len(texts)}  classes={n_classes}")

    if not os.path.isdir(args.checkpoint):
        msg = (
            f"Checkpoint directory not found: {args.checkpoint}. The wiring "
            "is in place but a real saved student is required to run the "
            "attack. See the README extension section for what each "
            "checkpoint directory must contain."
        )
        logger.log_summary({"status": "missing_checkpoint", "message": msg})
        logger.close()
        print("\n" + msg)
        sys.exit(2)

    print("\nLoading student...")
    from robustness.students import load_student
    student = load_student(args.student, args.checkpoint, dataset=args.dataset)
    if student.n_classes != n_classes:
        print(
            f"  warning: student.n_classes={student.n_classes} but "
            f"dataset has {n_classes} classes."
        )

    if args.dry_run:
        print("\n--dry_run set; skipping attack.")
        logger.log_summary({"status": "dry_run_ok"})
        logger.close()
        return

    # Reuse the existing TextFooler / BERT-Attack runners. They speak
    # the same wrapper interface as the drop-clause TM, so the student
    # plugs in unchanged.
    from robustness.attack_runner import run_textfooler_attack, run_bert_attack

    # Filter the test set down to correctly-classified samples so the
    # attack budget is spent on a fair starting point.
    print("\nFiltering to correctly-classified test samples...")
    t0 = time.time()
    correct_indices = []
    batch = 32
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        logits = student.predict_logits(chunk)
        preds = np.argmax(logits, axis=1)
        for j, p in enumerate(preds):
            if int(p) == labels[i + j]:
                correct_indices.append(i + j)
        if len(correct_indices) >= args.max_samples:
            break
    print(
        f"  found {len(correct_indices)} correct samples in "
        f"{time.time() - t0:.1f}s"
    )
    if not correct_indices:
        logger.log_summary({"status": "no_correct_samples"})
        logger.close()
        print("Student got every sample wrong; nothing to attack.")
        sys.exit(3)

    attack_indices = correct_indices[:args.max_samples]
    attack_data = [(texts[i], int(labels[i])) for i in attack_indices]

    print(f"\nRunning {args.recipe} on {len(attack_data)} samples...")
    if args.recipe == "textfooler":
        results = run_textfooler_attack(student, attack_data,
                                        max_samples=len(attack_data),
                                        seed=args.seed)
    elif args.recipe == "bert_attack":
        results = run_bert_attack(student, attack_data,
                                  max_samples=len(attack_data),
                                  seed=args.seed)
    else:
        raise ValueError(f"Unknown recipe: {args.recipe}")

    summary = {
        "status": "ok",
        "student": args.student,
        "dataset": args.dataset,
        "recipe": args.recipe,
        "n_correct": len(correct_indices),
        "n_attacked": len(attack_data),
        "adversarial_accuracy": results.get("adversarial_accuracy"),
        "attack_success_rate": results.get("attack_success_rate"),
        "attack_time": results.get("attack_time"),
    }
    logger.log_summary(summary)
    logger.close()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "per_sample.json"), "w") as f:
        json.dump(results.get("per_sample_results", []), f, indent=2)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
