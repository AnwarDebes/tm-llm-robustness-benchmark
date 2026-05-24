"""Student: Paper A subword + dependency GraphTM.

Loads a trained ``MultiClassGraphTsetlinMachine`` from paper A. The
expected checkpoint directory contains:

    tm_state.pkl   pickle saved by ``tm.save(fname)``; the GraphTM state
                   dict including ``ta_state``, ``clause_weights``,
                   ``hypervectors``, and TM configuration.
    vocab.json     list[str] of subword tokens used as the GraphTM
                   symbol space. Order matters (it is the order used at
                   training time).
    labels.json    {"label_names": [...], "n_classes": int}.
    config.json    arguments captured from ``train_paper_a_subword_dep
                   _graphtm.py``: dataset, hv_size, hv_bits, max_subwords,
                   depth, message_size, message_bits, clauses, T, s,
                   max_included_literals, use_sequential_edges,
                   use_dep_edges.

At inference, attacked text is re-graphed using the same
``SubwordDepGraphBuilder`` configuration. GraphTM symbol encoding is
re-initialised against the saved vocabulary; the test ``Graphs`` is
built with ``init_with`` set to a one-doc reference graph so that the
hypervector layout matches what was used during training.

This module never trains anything: it only rebuilds the graph and
calls ``score`` and ``predict`` on the loaded TM.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import numpy as np

from .base import GraphTMStudent


def _ensure_paper_a_on_path():
    """Add paper-a's src/ to sys.path if not already importable."""
    try:
        import graphtm.subword_dep_graph  # noqa: F401
        return
    except ImportError:
        pass
    candidates = [
        os.environ.get("PAPER_A_ROOT", ""),
        os.path.expanduser("~/project/paper-a-subword-dep-graphtm"),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "..", "paper-a-subword-dep-graphtm"
            )
        ),
    ]
    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "src", "graphtm")):
            sys.path.insert(0, os.path.join(root, "src"))
            return
    raise ImportError(
        "Could not locate paper-a-subword-dep-graphtm/src on sys.path. "
        "Set PAPER_A_ROOT or check the sibling project layout."
    )


class SubwordDepGraphTMStudent(GraphTMStudent):
    """Wraps a saved Paper-A GraphTM for TextAttack."""

    family = "subword_dep_graphtm"

    def __init__(self, tm, builder, vocab, label_names, n_classes, hv_size, hv_bits,
                 reference_graphs):
        self.model = tm
        self.builder = builder
        self.vocab = list(vocab)
        self.label_names = list(label_names)
        self.n_classes = int(n_classes)
        self._hv_size = hv_size
        self._hv_bits = hv_bits
        # Reference graphs object holding the training-time symbol encoding.
        # Test-time graphs are built with init_with=reference_graphs.
        self._reference_graphs = reference_graphs

    def predict_logits(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        # Rebuild a Graphs object covering the input batch.
        graphs = self.builder.build_graphtm_graphs(
            list(texts),
            self.vocab,
            hypervector_size=self._hv_size,
            hypervector_bits=self._hv_bits,
            init_with=self._reference_graphs,
        )
        # GraphTM exposes score(graphs) -> ndarray (n, n_classes). Use it
        # rather than predict() so we get the real-valued vote margins.
        if hasattr(self.model, "score"):
            raw = self.model.score(graphs)
        else:
            preds = self.model.predict(graphs)
            raw = np.zeros((len(preds), self.n_classes), dtype=np.float32)
            for i, p in enumerate(preds):
                raw[i, int(p)] = 1.0
        return self._scores_to_logits(raw)


def load(checkpoint_dir, dataset=None, reference_text=None, mock_tm=None,
         mock_builder=None, mock_reference_graphs=None):
    """Load a Paper-A subword-dependency GraphTM student.

    Args:
        checkpoint_dir: directory with ``tm_state.pkl``, ``vocab.json``,
            ``labels.json``, ``config.json``. Ignored when ``mock_tm``
            is supplied.
        dataset: dataset tag for logging. Optional; the labels come from
            the checkpoint, not the dataset name.
        reference_text: optional short string used to build the
            reference ``Graphs`` object that holds the training-time HV
            encoding. If None, a single-token placeholder is used.
        mock_tm: test-only injection. When set, ``checkpoint_dir`` is
            not read and ``mock_tm`` is used as the TM directly.
        mock_builder: test-only injection. When set with ``mock_tm``,
            this builder is used in place of ``SubwordDepGraphBuilder``.
        mock_reference_graphs: test-only injection for the reference
            graphs object.
    """
    if mock_tm is not None:
        if mock_builder is None or mock_reference_graphs is None:
            raise ValueError(
                "mock_tm requires mock_builder and mock_reference_graphs."
            )
        return SubwordDepGraphTMStudent(
            tm=mock_tm,
            builder=mock_builder,
            vocab=getattr(mock_builder, "vocab", ["[UNK]"]),
            label_names=getattr(mock_tm, "label_names", ["0", "1"]),
            n_classes=getattr(mock_tm, "n_classes", 2),
            hv_size=getattr(mock_tm, "hv_size", 64),
            hv_bits=getattr(mock_tm, "hv_bits", 2),
            reference_graphs=mock_reference_graphs,
        )

    _ensure_paper_a_on_path()
    from graphtm.subword_dep_graph import SubwordDepGraphBuilder
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(checkpoint_dir, "vocab.json")) as f:
        vocab = json.load(f)
    with open(os.path.join(checkpoint_dir, "labels.json")) as f:
        labels_meta = json.load(f)

    builder = SubwordDepGraphBuilder(
        tokenizer_name=cfg.get("tokenizer_name", "bert-base-uncased"),
        spacy_model=cfg.get("spacy_model", "en_core_web_sm"),
        max_subwords=cfg.get("max_subwords", 128),
        use_sequential_edges=cfg.get("use_sequential_edges", True),
        use_dep_edges=cfg.get("use_dep_edges", True),
    )

    # Reference graphs: a one-document Graphs object holding the
    # training-time hypervector encoding for the symbol space.
    ref_text = reference_text if reference_text is not None else vocab[0]
    reference_graphs = builder.build_graphtm_graphs(
        [ref_text], vocab,
        hypervector_size=cfg.get("hv_size", 1024),
        hypervector_bits=cfg.get("hv_bits", 4),
    )

    tm = MultiClassGraphTsetlinMachine(
        cfg["clauses"], cfg["T"], cfg["s"],
        depth=cfg.get("depth", 2),
        message_size=cfg.get("message_size", 512),
        message_bits=cfg.get("message_bits", 4),
        max_included_literals=cfg.get("max_included_literals", 32),
    )
    tm.load(fname=os.path.join(checkpoint_dir, "tm_state.pkl"))

    return SubwordDepGraphTMStudent(
        tm=tm,
        builder=builder,
        vocab=vocab,
        label_names=labels_meta["label_names"],
        n_classes=labels_meta["n_classes"],
        hv_size=cfg.get("hv_size", 1024),
        hv_bits=cfg.get("hv_bits", 4),
        reference_graphs=reference_graphs,
    )


__all__ = ["SubwordDepGraphTMStudent", "load"]
