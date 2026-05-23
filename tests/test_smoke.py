"""Smoke tests: verify the robustness wrapper imports and counterfactual data is present."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_tm_wrapper_imports():
    from robustness.tm_wrapper import TMModelWrapper  # noqa: F401


def test_attack_runner_imports():
    from robustness.attack_runner import run_textfooler_attack  # noqa: F401


def test_eval_logger_imports():
    from eval.logger import ExperimentLogger  # noqa: F401


def test_counterfactual_imdb_present():
    p = ROOT / "data" / "robustness" / "counterfactual_imdb.json"
    assert p.exists(), "counterfactual_imdb.json missing"
    with open(p) as f:
        data = json.load(f)
    assert len(data) > 0
