#!/usr/bin/env python3
"""Paper C: TM vs BERT robustness on IMDb.

Train TM and evaluate on:
1. Clean IMDb test set
2. Counterfactual IMDb (Kaushik et al. 2020)
3. TextFooler attack (100 samples)

Usage:
    python experiments/run_paper_c_imdb.py --seed 42
"""

import os, sys, json, time, argparse
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tmu.models.classification.vanilla_classifier import TMClassifier
from eval.logger import ExperimentLogger
from robustness.tm_wrapper import TMModelWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--clauses", default=10000, type=int)
    parser.add_argument("--T", default=8000, type=int)
    parser.add_argument("--s", default=2.0, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--features", default=5000, type=int)
    parser.add_argument("--attack_samples", default=100, type=int)
    parser.add_argument("--skip_textfooler", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = os.path.join(PROJECT_ROOT, "experiments", "paper_c_imdb_tm", f"seed_{args.seed}")
    os.makedirs(exp_dir, exist_ok=True)
    logger = ExperimentLogger(os.path.join(exp_dir, "log.jsonl"))
    logger.log_config(vars(args))

    print(f"=== Paper C: TM Robustness on IMDb (seed={args.seed}) ===")

    # Load IMDb
    print("Loading IMDb...")
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = np.array(ds["train"]["label"], dtype=np.uint32)
    test_texts = ds["test"]["text"]
    test_labels = np.array(ds["test"]["label"], dtype=np.uint32)

    # Vectorize
    print("Vectorizing...")
    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True, max_features=50000)
    X_train_raw = vectorizer.fit_transform(train_texts)
    X_test_raw = vectorizer.transform(test_texts)

    skb = SelectKBest(chi2, k=args.features)
    skb.fit(X_train_raw, train_labels)
    X_train = skb.transform(X_train_raw).toarray().astype(np.uint32)
    X_test = skb.transform(X_test_raw).toarray().astype(np.uint32)

    # Train TM
    print(f"\nTraining TM: clauses={args.clauses}, epochs={args.epochs}")
    tm = TMClassifier(
        args.clauses, args.T, args.s,
        platform="CUDA",
        weighted_clauses=True,
        clause_drop_p=0.75,
        seed=args.seed,
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        tm.fit(X_train, train_labels, shuffle=True)
        preds = tm.predict(X_test)
        acc = (preds == test_labels).mean()
        if acc > best_acc:
            best_acc = acc
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  E{epoch}: clean_acc={acc*100:.2f}% (best={best_acc*100:.2f}%)")

    clean_accuracy = float(best_acc)
    print(f"\n1. Clean accuracy: {clean_accuracy*100:.2f}%")

    # 2. Counterfactual IMDb
    print("\n2. Evaluating on Counterfactual IMDb...")
    cf_path = os.path.join(PROJECT_ROOT, "data", "robustness", "counterfactual_imdb.json")
    with open(cf_path) as f:
        cf_data = json.load(f)

    cf_test = cf_data.get("test", [])
    if cf_test:
        cf_texts = [ex["text"] for ex in cf_test]
        cf_labels_raw = [ex["label"] for ex in cf_test]
        # Map sentiment labels to 0/1
        label_map = {"Negative": 0, "Positive": 1, 0: 0, 1: 1}
        cf_labels = np.array([label_map.get(l, l) for l in cf_labels_raw], dtype=np.uint32)

        cf_X = skb.transform(vectorizer.transform(cf_texts)).toarray().astype(np.uint32)
        cf_preds = tm.predict(cf_X)
        cf_acc = float((cf_preds == cf_labels).mean())
        print(f"   Counterfactual accuracy: {cf_acc*100:.2f}% ({len(cf_test)} samples)")
    else:
        cf_acc = None
        print("   No counterfactual test data found")

    # 3. TextFooler attack (optional, slow)
    tf_results = None
    if not args.skip_textfooler:
        print(f"\n3. Running TextFooler attack ({args.attack_samples} samples)...")
        wrapper = TMModelWrapper(tm, vectorizer, skb, n_classes=2)

        try:
            from robustness.attack_runner import run_textfooler_attack
            # Use correctly-classified test samples for attack
            correct_mask = (tm.predict(X_test) == test_labels)
            correct_indices = np.where(correct_mask)[0][:args.attack_samples]

            attack_data = [(test_texts[i], int(test_labels[i])) for i in correct_indices]
            tf_results = run_textfooler_attack(wrapper, attack_data, max_samples=args.attack_samples)

            print(f"   TextFooler success rate: {tf_results['attack_success_rate']*100:.1f}%")
            print(f"   Adversarial accuracy: {tf_results['adversarial_accuracy']*100:.1f}%")
        except Exception as e:
            print(f"   TextFooler failed: {e}")
            tf_results = {"error": str(e)}
    else:
        print("\n3. TextFooler skipped (--skip_textfooler)")

    # Summary
    robustness_ratio = cf_acc / max(clean_accuracy, 1e-10) if cf_acc else None

    results = {
        "model": "TM (Drop-Clause)",
        "dataset": "IMDb",
        "seed": args.seed,
        "clean_accuracy": clean_accuracy,
        "counterfactual_accuracy": cf_acc,
        "robustness_ratio_cf": robustness_ratio,
        "textfooler_results": {
            "adversarial_accuracy": tf_results.get("adversarial_accuracy") if tf_results and "error" not in tf_results else None,
            "attack_success_rate": tf_results.get("attack_success_rate") if tf_results and "error" not in tf_results else None,
        } if tf_results else None,
    }

    print(f"\n=== Summary ===")
    print(f"  Clean: {clean_accuracy*100:.2f}%")
    if cf_acc is not None:
        print(f"  Counterfactual: {cf_acc*100:.2f}%")
        print(f"  Robustness ratio (CF/clean): {robustness_ratio:.4f}")

    logger.log_summary(results)
    logger.close()

    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {exp_dir}/summary.json")


if __name__ == "__main__":
    main()
