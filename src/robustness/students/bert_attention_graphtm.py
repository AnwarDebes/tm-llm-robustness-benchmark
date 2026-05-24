"""Student: Paper B BERT-attention GraphTM.

Loads the BERT-attention-distilled GraphTM from paper B. The student
needs three pieces:

1. A BERT teacher checkpoint (only used to extract per-document
   attention edges at inference time).
2. A trained ``MultiClassGraphTsetlinMachine`` saved with
   ``tm.save(fname)``.
3. The training-time vocabulary and label list.

Expected checkpoint directory layout:

    tm_state.pkl   GraphTM state dict.
    vocab.json     list[str], the symbol vocabulary.
    labels.json    {"label_names": [...], "n_classes": int}.
    config.json    teacher_path, tokenizer_name, top_k, layers, max_len,
                   hv_size, hv_bits, clauses, T, s, depth, message_size,
                   message_bits, max_included_literals.

At inference, ``predict_logits`` reruns the same attention extraction
on the attacked text and rebuilds graphs using paper-b's
``build_graphs`` helper, then calls ``tm.score``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import numpy as np

from .base import GraphTMStudent


def _ensure_paper_b_on_path():
    candidates = [
        os.environ.get("PAPER_B_ROOT", ""),
        os.path.expanduser("~/project/paper-b-attention-distill-graphtm"),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "..", "paper-b-attention-distill-graphtm"
            )
        ),
    ]
    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "experiments")):
            sys.path.insert(0, root)
            return root
    raise ImportError(
        "Could not locate paper-b-attention-distill-graphtm. "
        "Set PAPER_B_ROOT or check the sibling project layout."
    )


class BertAttentionGraphTMStudent(GraphTMStudent):
    """Wraps a saved Paper-B BERT-attention GraphTM for TextAttack."""

    family = "bert_attention_graphtm"

    def __init__(self, tm, extract_fn, build_graphs_fn, vocab, label_names,
                 n_classes, hv_size, hv_bits, teacher_path, tokenizer_name,
                 top_k, layers, max_len, reference_graphs):
        self.model = tm
        self._extract_fn = extract_fn
        self._build_graphs_fn = build_graphs_fn
        self.vocab = list(vocab)
        self.label_names = list(label_names)
        self.n_classes = int(n_classes)
        self._hv_size = hv_size
        self._hv_bits = hv_bits
        self._teacher_path = teacher_path
        self._tokenizer_name = tokenizer_name
        self._top_k = top_k
        self._layers = list(layers) if layers is not None else [6, 8, 10]
        self._max_len = max_len
        self._reference_graphs = reference_graphs

    def predict_logits(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        # Re-extract attention graphs for the attacked batch.
        nodes, edges, _ = self._extract_fn(
            list(texts),
            np.zeros(len(texts), dtype=np.int64),
            self._teacher_path,
            self._tokenizer_name,
            top_k=self._top_k,
            layers=self._layers,
            max_len=self._max_len,
        )
        graphs = self._build_graphs_fn(
            nodes, edges, self.vocab,
            hv_size=self._hv_size, hv_bits=self._hv_bits,
            init_with=self._reference_graphs,
        )
        if hasattr(self.model, "score"):
            raw = self.model.score(graphs)
        else:
            preds = self.model.predict(graphs)
            raw = np.zeros((len(preds), self.n_classes), dtype=np.float32)
            for i, p in enumerate(preds):
                raw[i, int(p)] = 1.0
        return self._scores_to_logits(raw)


def load(checkpoint_dir, dataset=None, reference_text=None, mock_tm=None,
         mock_extract_fn=None, mock_build_graphs_fn=None,
         mock_reference_graphs=None):
    """Load a Paper-B BERT-attention GraphTM student.

    Args:
        checkpoint_dir: directory with ``tm_state.pkl``, ``vocab.json``,
            ``labels.json``, ``config.json``. Ignored when ``mock_tm`` is
            supplied.
        dataset: dataset tag for logging.
        reference_text: optional short string used to seed the reference
            ``Graphs`` object for symbol encoding consistency.
        mock_tm, mock_extract_fn, mock_build_graphs_fn,
        mock_reference_graphs: test-only injection points so the student
            can be exercised without a real saved GraphTM or BERT
            teacher.
    """
    if mock_tm is not None:
        if (mock_extract_fn is None or mock_build_graphs_fn is None
                or mock_reference_graphs is None):
            raise ValueError(
                "mock_tm requires mock_extract_fn, mock_build_graphs_fn, "
                "mock_reference_graphs."
            )
        return BertAttentionGraphTMStudent(
            tm=mock_tm,
            extract_fn=mock_extract_fn,
            build_graphs_fn=mock_build_graphs_fn,
            vocab=getattr(mock_tm, "vocab", ["[UNK]"]),
            label_names=getattr(mock_tm, "label_names", ["0", "1"]),
            n_classes=getattr(mock_tm, "n_classes", 2),
            hv_size=getattr(mock_tm, "hv_size", 64),
            hv_bits=getattr(mock_tm, "hv_bits", 2),
            teacher_path=getattr(mock_tm, "teacher_path", "MOCK"),
            tokenizer_name=getattr(mock_tm, "tokenizer_name", "bert-base-uncased"),
            top_k=getattr(mock_tm, "top_k", 5),
            layers=getattr(mock_tm, "layers", [6, 8, 10]),
            max_len=getattr(mock_tm, "max_len", 128),
            reference_graphs=mock_reference_graphs,
        )

    _ensure_paper_b_on_path()
    # Paper-B's helpers live inside the train script. Import them
    # directly so paper-c does not duplicate the extraction logic.
    from experiments.train_paper_b_attention_distill import (
        extract_attention_graphs as _extract,
        build_graphs as _build_graphs,
    )
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(checkpoint_dir, "vocab.json")) as f:
        vocab = json.load(f)
    with open(os.path.join(checkpoint_dir, "labels.json")) as f:
        labels_meta = json.load(f)

    # Reference graphs: one-doc Graphs in the training-time HV layout.
    ref_text = reference_text if reference_text is not None else vocab[0]
    ref_nodes, ref_edges, _ = _extract(
        [ref_text],
        np.zeros(1, dtype=np.int64),
        cfg["teacher_path"],
        cfg.get("tokenizer_name", "bert-base-uncased"),
        top_k=cfg.get("top_k", 5),
        layers=cfg.get("layers", [6, 8, 10]),
        max_len=cfg.get("max_len", 128),
    )
    reference_graphs = _build_graphs(
        ref_nodes, ref_edges, vocab,
        hv_size=cfg.get("hv_size", 512),
        hv_bits=cfg.get("hv_bits", 2),
    )

    tm = MultiClassGraphTsetlinMachine(
        cfg["clauses"], cfg["T"], cfg["s"],
        depth=cfg.get("depth", 1),
        message_size=cfg.get("message_size", 256),
        message_bits=cfg.get("message_bits", 2),
        max_included_literals=cfg.get("max_included_literals", 32),
    )
    tm.load(fname=os.path.join(checkpoint_dir, "tm_state.pkl"))

    return BertAttentionGraphTMStudent(
        tm=tm,
        extract_fn=_extract,
        build_graphs_fn=_build_graphs,
        vocab=vocab,
        label_names=labels_meta["label_names"],
        n_classes=labels_meta["n_classes"],
        hv_size=cfg.get("hv_size", 512),
        hv_bits=cfg.get("hv_bits", 2),
        teacher_path=cfg["teacher_path"],
        tokenizer_name=cfg.get("tokenizer_name", "bert-base-uncased"),
        top_k=cfg.get("top_k", 5),
        layers=cfg.get("layers", [6, 8, 10]),
        max_len=cfg.get("max_len", 128),
        reference_graphs=reference_graphs,
    )


__all__ = ["BertAttentionGraphTMStudent", "load"]
