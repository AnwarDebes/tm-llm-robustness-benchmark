#!/usr/bin/env python3
"""Unified attack harness for comparing TM and transformer robustness.

Runs TextFooler and BERT-Attack against any classifier that exposes
a predict interface. Uses TextAttack library.

Paper C methodology:
- Standard attack budget: <=10% word replacement
- Report: clean accuracy, adversarial accuracy, robustness ratio (adv/clean)
- All results logged per-sample for downstream analysis
"""

import os
import sys
import json
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_textfooler_attack(model_wrapper, dataset, max_samples=500, seed=42):
    """Run TextFooler attack against a wrapped model.

    Args:
        model_wrapper: TextAttack-compatible model wrapper
        dataset: list of (text, label) tuples
        max_samples: max number of samples to attack
        seed: random seed

    Returns:
        dict with attack results
    """
    import textattack
    from textattack.attack_recipes import TextFoolerJin2019
    from textattack.datasets import Dataset as TADataset

    # Build TextAttack dataset
    ta_dataset = TADataset(
        [(text, label) for text, label in dataset[:max_samples]],
        input_columns=["text"],
        label_map=None,
    )

    # Build attack
    attack = TextFoolerJin2019.build(model_wrapper)

    # Run attack
    results = []
    n_success = 0
    n_failed = 0
    n_skipped = 0

    attack_args = textattack.AttackArgs(
        num_examples=min(max_samples, len(dataset)),
        random_seed=seed,
        disable_stdout=True,
    )

    attacker = textattack.Attacker(attack, ta_dataset, attack_args)

    print(f"Running TextFooler on {min(max_samples, len(dataset))} samples...")
    t0 = time.time()

    for result in attacker.attack_dataset():
        entry = {
            "original_text": result.original_result.attacked_text.text if hasattr(result, 'original_result') else "",
            "original_label": int(result.original_result.ground_truth_output) if hasattr(result, 'original_result') else -1,
        }

        if isinstance(result, textattack.attack_results.SuccessfulAttackResult):
            n_success += 1
            entry["attack_success"] = True
            entry["perturbed_text"] = result.perturbed_result.attacked_text.text
        elif isinstance(result, textattack.attack_results.FailedAttackResult):
            n_failed += 1
            entry["attack_success"] = False
        else:
            n_skipped += 1
            entry["attack_success"] = None  # skipped

        results.append(entry)

    attack_time = time.time() - t0
    total = n_success + n_failed + n_skipped

    return {
        "attack": "TextFooler",
        "total_samples": total,
        "successful_attacks": n_success,
        "failed_attacks": n_failed,
        "skipped": n_skipped,
        "attack_success_rate": n_success / max(total - n_skipped, 1),
        "adversarial_accuracy": n_failed / max(total - n_skipped, 1),
        "attack_time": attack_time,
        "per_sample_results": results,
    }


def run_bert_attack(model_wrapper, dataset, max_samples=500, seed=42):
    """Run BERT-Attack against a wrapped model."""
    import textattack
    from textattack.attack_recipes import BERTAttackLi2020
    from textattack.datasets import Dataset as TADataset

    ta_dataset = TADataset(
        [(text, label) for text, label in dataset[:max_samples]],
        input_columns=["text"],
        label_map=None,
    )

    attack = BERTAttackLi2020.build(model_wrapper)

    n_success = 0
    n_failed = 0
    n_skipped = 0
    results = []

    attack_args = textattack.AttackArgs(
        num_examples=min(max_samples, len(dataset)),
        random_seed=seed,
        disable_stdout=True,
    )

    attacker = textattack.Attacker(attack, ta_dataset, attack_args)

    print(f"Running BERT-Attack on {min(max_samples, len(dataset))} samples...")
    t0 = time.time()

    for result in attacker.attack_dataset():
        entry = {}
        if isinstance(result, textattack.attack_results.SuccessfulAttackResult):
            n_success += 1
            entry["attack_success"] = True
        elif isinstance(result, textattack.attack_results.FailedAttackResult):
            n_failed += 1
            entry["attack_success"] = False
        else:
            n_skipped += 1
            entry["attack_success"] = None
        results.append(entry)

    attack_time = time.time() - t0
    total = n_success + n_failed + n_skipped

    return {
        "attack": "BERT-Attack",
        "total_samples": total,
        "successful_attacks": n_success,
        "failed_attacks": n_failed,
        "skipped": n_skipped,
        "attack_success_rate": n_success / max(total - n_skipped, 1),
        "adversarial_accuracy": n_failed / max(total - n_skipped, 1),
        "attack_time": attack_time,
        "per_sample_results": results,
    }
