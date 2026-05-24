"""Synthetic self-tests for the Tsetlin-family extension.

These tests do not require a real saved GraphTM student or any
sibling-project import. They use the ``mock_*`` injection points in
each student's ``load`` function to plug in deterministic stand-ins for
the graph builder, attention teacher, and the GraphTM itself, then
exercise the public ``predict_logits`` and ``__call__`` surface.

The point of the tests is to lock the API contract so future students
cannot drift it. They verify:

- predict_logits returns an ndarray of shape (n, n_classes), dtype
  float32, for a batch of three sample sentences.
- __call__ accepts both a single string and a list and returns the
  same shape.
- A gradient request (``backward``) raises NotImplementedError with a
  clear message.
- Empty input returns an empty (0, n_classes) ndarray rather than
  crashing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


SAMPLE_TEXTS = [
    "the stock market closed slightly higher today.",
    "wheat futures fell sharply on weather concerns.",
    "central bank held rates steady at the meeting.",
]


# Deterministic mock GraphTM that returns vote sums depending on the
# number of input graphs. It implements both ``score`` and ``predict``
# so we can verify the student prefers ``score``.
class MockGraphTM:
    def __init__(self, n_classes=3, vocab=None, label_names=None):
        self.n_classes = n_classes
        self.vocab = vocab or ["[UNK]"]
        self.label_names = label_names or [str(i) for i in range(n_classes)]
        # Other config the students read off the mock for the no-real-load path.
        self.hv_size = 64
        self.hv_bits = 2

    def score(self, graphs):
        n = getattr(graphs, "num_graphs", 1)
        # Deterministic vote pattern: class i gets vote (i + 1) * n.
        out = np.zeros((n, self.n_classes), dtype=np.int32)
        for i in range(n):
            for c in range(self.n_classes):
                out[i, c] = (c + 1) * (i + 1)
        return out

    def predict(self, graphs):
        return np.argmax(self.score(graphs), axis=1)


class MockGraphs:
    """Stand-in for GraphTsetlinMachine.graphs.Graphs."""
    def __init__(self, num_graphs):
        self.num_graphs = num_graphs


class MockSubwordBuilder:
    """Stand-in for paper-a's SubwordDepGraphBuilder. Provides only the
    one method the student actually calls."""
    vocab = ["[UNK]", "the", "stock", "market", "wheat", "futures",
            "central", "bank", "rates"]

    def build_graphtm_graphs(self, texts, vocab_symbols,
                             hypervector_size=64, hypervector_bits=2,
                             init_with=None):
        return MockGraphs(len(texts))


def _mock_extract_fn(texts, labels, model_path, tokenizer_name,
                    top_k=5, layers=None, max_len=128, batch_size=32):
    """Stand-in for paper-b's extract_attention_graphs.

    Returns nodes / edges / vocab in the format the real function uses.
    """
    nodes = []
    edges = []
    for t in texts:
        tokens = t.split()
        node_list = [(f"{tok}_{i}", tok) for i, tok in enumerate(tokens)] or [("EMPTY_0", "EMPTY")]
        edge_list = []
        for i in range(len(node_list) - 1):
            edge_list.append((node_list[i][0], node_list[i + 1][0], "attn"))
        nodes.append(node_list)
        edges.append(edge_list)
    vocab = list({tok for n in nodes for _, tok in n})
    return nodes, edges, vocab


def _mock_build_graphs_fn(doc_nodes, doc_edges, vocab,
                          hv_size=64, hv_bits=2, init_with=None):
    return MockGraphs(len(doc_nodes))


class MockQwenTeacher:
    """Stand-in for QwenTeacher: only ``extract_attention_graphs`` is called."""
    def extract_attention_graphs(self, texts, top_k=5, layers=None,
                                 max_len=128, batch_size=8,
                                 vocab_size=5000, verbose=False):
        return _mock_extract_fn(texts, None, None, None)


# Common: each student exposes the same surface.
@pytest.fixture(params=["subword_dep", "bert_attention", "qwen_attention"])
def student(request):
    family = request.param
    if family == "subword_dep":
        from robustness.students import subword_dep_graphtm
        return subword_dep_graphtm.load(
            checkpoint_dir="UNUSED_WITH_MOCK",
            mock_tm=MockGraphTM(),
            mock_builder=MockSubwordBuilder(),
            mock_reference_graphs=MockGraphs(1),
        )
    if family == "bert_attention":
        from robustness.students import bert_attention_graphtm
        return bert_attention_graphtm.load(
            checkpoint_dir="UNUSED_WITH_MOCK",
            mock_tm=MockGraphTM(),
            mock_extract_fn=_mock_extract_fn,
            mock_build_graphs_fn=_mock_build_graphs_fn,
            mock_reference_graphs=MockGraphs(1),
        )
    if family == "qwen_attention":
        from robustness.students import qwen_attention_graphtm
        return qwen_attention_graphtm.load(
            checkpoint_dir="UNUSED_WITH_MOCK",
            mock_tm=MockGraphTM(),
            mock_teacher=MockQwenTeacher(),
            mock_build_graphs_fn=_mock_build_graphs_fn,
            mock_reference_graphs=MockGraphs(1),
        )
    raise AssertionError(family)


def test_predict_logits_shape(student):
    logits = student.predict_logits(SAMPLE_TEXTS)
    assert isinstance(logits, np.ndarray), type(logits)
    assert logits.shape == (3, student.n_classes), logits.shape
    assert logits.dtype == np.float32


def test_call_accepts_list(student):
    out = student(SAMPLE_TEXTS)
    assert out.shape == (3, student.n_classes)


def test_call_accepts_single_string(student):
    out = student(SAMPLE_TEXTS[0])
    assert out.shape == (1, student.n_classes)


def test_empty_input(student):
    out = student.predict_logits([])
    assert out.shape == (0, student.n_classes)


def test_backward_raises(student):
    with pytest.raises(NotImplementedError):
        student.backward()


def test_family_label_is_set(student):
    assert isinstance(student.family, str) and student.family != "abstract"


def test_logits_have_signal(student):
    # Mock GraphTM returns different vote sums per input; the wrapper
    # must not collapse them.
    logits = student.predict_logits(SAMPLE_TEXTS)
    # Rows should differ for at least one column.
    deltas = (logits.max(axis=0) - logits.min(axis=0))
    assert (deltas > 0).any(), (
        "predict_logits returned identical scores for distinct inputs; "
        "the wrapper is collapsing the score matrix."
    )


def test_dispatcher_rejects_unknown_family():
    from robustness.students import load_student
    with pytest.raises(ValueError):
        load_student("not_a_real_family", "anywhere")


def test_n_classes_matches_label_names(student):
    assert len(student.label_names) == student.n_classes


def test_call_returns_float32_for_textattack(student):
    # TextAttack consumes float scores; make sure dtype is stable.
    out = student(SAMPLE_TEXTS)
    assert out.dtype == np.float32
