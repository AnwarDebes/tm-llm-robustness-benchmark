#!/usr/bin/env python3
"""Generate the results bar chart for the README.

Reads results/paper_c_imdb_robustness.json and writes figures/results.png.
Re-run after updating results to refresh the figure.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(REPO_ROOT / "results" / "paper_c_imdb_robustness.json") as f:
        results = json.load(f)

    tm = results["models"]["TM_DropClause"]
    bert = results["models"]["BERT_base"]

    models = ["TM (drop-clause)", "BERT-base"]
    clean = [tm["clean_accuracy"] * 100, bert["clean_accuracy"] * 100]
    counterfactual = [
        tm["counterfactual_accuracy"] * 100,
        bert["counterfactual_accuracy"] * 100,
    ]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=130)
    bars_clean = ax.bar(
        x - width / 2, clean, width,
        label="Clean IMDb test",
        color="#4C72B0", edgecolor="black", linewidth=0.6,
    )
    bars_cf = ax.bar(
        x + width / 2, counterfactual, width,
        label="Counterfactual IMDb (488 paired edits)",
        color="#DD8452", edgecolor="black", linewidth=0.6,
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Clean vs counterfactual accuracy on IMDb sentiment")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower center", frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in list(bars_clean) + list(bars_cf):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "results.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
