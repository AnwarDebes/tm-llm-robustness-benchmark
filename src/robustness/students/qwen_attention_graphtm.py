"""Student: decoder-attention-distill-graphtm Qwen-attention GraphTM.

Loads the Qwen2.5 attention-distilled GraphTM from the sibling project
``decoder-attention-distill-graphtm``. The sibling project already
provides:

  - ``src/teachers/qwen_attention.QwenTeacher.from_pretrained(model, adapter_path=...)``
    that exposes ``extract_attention_graphs(texts, top_k, layers, max_len)``.
  - ``src/graphs/builder.build_graphs(nodes, edges, vocab, ...)``.

This student reuses both helpers verbatim. The student itself only
holds the trained ``MultiClassGraphTsetlinMachine`` and reruns
extraction plus graph build at inference.

Expected checkpoint directory layout:

    tm_state.pkl   GraphTM state dict.
    vocab.json     list[str].
    labels.json    {"label_names": [...], "n_classes": int}.
    config.json    teacher_model (HF id), teacher_path (LoRA adapter
                   directory), top_k, layers, max_len, hv_size, hv_bits,
                   clauses, T, s, depth, message_size, message_bits,
                   max_included_literals, load_in_4bit.

Decoder LMs are large. The student loads the teacher once and keeps it
on the GPU; attack runners that go through many small batches benefit
from this. Set ``load_in_4bit: false`` in the config when the GPU does
not support bitsandbytes.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import numpy as np

from .base import GraphTMStudent


def _ensure_decoder_on_path():
    candidates = [
        os.environ.get("DECODER_DISTILL_ROOT", ""),
        os.path.expanduser("~/project/decoder-attention-distill-graphtm"),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "..", "decoder-attention-distill-graphtm"
            )
        ),
    ]
    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "src", "teachers")):
            sys.path.insert(0, os.path.join(root, "src"))
            return root
    raise ImportError(
        "Could not locate decoder-attention-distill-graphtm/src. "
        "Set DECODER_DISTILL_ROOT or check the sibling project layout."
    )


class QwenAttentionGraphTMStudent(GraphTMStudent):
    """Wraps a saved Qwen-attention GraphTM for TextAttack."""

    family = "qwen_attention_graphtm"

    def __init__(self, tm, teacher, build_graphs_fn, vocab, label_names,
                 n_classes, hv_size, hv_bits, top_k, layers, max_len,
                 extract_batch_size, vocab_size, reference_graphs):
        self.model = tm
        self.teacher = teacher
        self._build_graphs_fn = build_graphs_fn
        self.vocab = list(vocab)
        self.label_names = list(label_names)
        self.n_classes = int(n_classes)
        self._hv_size = hv_size
        self._hv_bits = hv_bits
        self._top_k = top_k
        self._layers = list(layers) if layers is not None else None
        self._max_len = max_len
        self._extract_batch_size = extract_batch_size
        self._vocab_size = vocab_size
        self._reference_graphs = reference_graphs

    def predict_logits(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        nodes, edges, _ = self.teacher.extract_attention_graphs(
            list(texts),
            top_k=self._top_k,
            layers=self._layers,
            max_len=self._max_len,
            batch_size=self._extract_batch_size,
            vocab_size=self._vocab_size,
            verbose=False,
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
         mock_teacher=None, mock_build_graphs_fn=None,
         mock_reference_graphs=None):
    """Load a decoder-attention-distill Qwen GraphTM student.

    Args:
        checkpoint_dir: directory with ``tm_state.pkl``, ``vocab.json``,
            ``labels.json``, ``config.json``.
        dataset: dataset tag for logging.
        reference_text: optional short string for the reference Graphs.
        mock_tm, mock_teacher, mock_build_graphs_fn, mock_reference_graphs:
            test-only injection points.
    """
    if mock_tm is not None:
        if (mock_teacher is None or mock_build_graphs_fn is None
                or mock_reference_graphs is None):
            raise ValueError(
                "mock_tm requires mock_teacher, mock_build_graphs_fn, "
                "mock_reference_graphs."
            )
        return QwenAttentionGraphTMStudent(
            tm=mock_tm,
            teacher=mock_teacher,
            build_graphs_fn=mock_build_graphs_fn,
            vocab=getattr(mock_tm, "vocab", ["[UNK]"]),
            label_names=getattr(mock_tm, "label_names", ["0", "1"]),
            n_classes=getattr(mock_tm, "n_classes", 2),
            hv_size=getattr(mock_tm, "hv_size", 64),
            hv_bits=getattr(mock_tm, "hv_bits", 2),
            top_k=getattr(mock_tm, "top_k", 5),
            layers=getattr(mock_tm, "layers", None),
            max_len=getattr(mock_tm, "max_len", 128),
            extract_batch_size=getattr(mock_tm, "extract_batch_size", 8),
            vocab_size=getattr(mock_tm, "vocab_size", 5000),
            reference_graphs=mock_reference_graphs,
        )

    _ensure_decoder_on_path()
    from teachers.qwen_attention import QwenTeacher
    from graphs.builder import build_graphs as _build_graphs
    from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(checkpoint_dir, "vocab.json")) as f:
        vocab = json.load(f)
    with open(os.path.join(checkpoint_dir, "labels.json")) as f:
        labels_meta = json.load(f)

    teacher = QwenTeacher.from_pretrained(
        cfg["teacher_model"],
        adapter_path=cfg["teacher_path"],
        num_labels=labels_meta["n_classes"],
        load_in_4bit=cfg.get("load_in_4bit", False),
    )

    ref_text = reference_text if reference_text is not None else vocab[0]
    ref_nodes, ref_edges, _ = teacher.extract_attention_graphs(
        [ref_text],
        top_k=cfg.get("top_k", 5),
        layers=cfg.get("layers"),
        max_len=cfg.get("max_len", 128),
        batch_size=1,
        vocab_size=cfg.get("vocab_size", 5000),
        verbose=False,
    )
    reference_graphs = _build_graphs(
        ref_nodes, ref_edges, vocab,
        hv_size=cfg.get("hv_size", 1024),
        hv_bits=cfg.get("hv_bits", 4),
    )

    tm = MultiClassGraphTsetlinMachine(
        cfg["clauses"], cfg["T"], cfg["s"],
        depth=cfg.get("depth", 2),
        message_size=cfg.get("message_size", 512),
        message_bits=cfg.get("message_bits", 4),
        max_included_literals=cfg.get("max_included_literals", 32),
    )
    tm.load(fname=os.path.join(checkpoint_dir, "tm_state.pkl"))

    return QwenAttentionGraphTMStudent(
        tm=tm,
        teacher=teacher,
        build_graphs_fn=_build_graphs,
        vocab=vocab,
        label_names=labels_meta["label_names"],
        n_classes=labels_meta["n_classes"],
        hv_size=cfg.get("hv_size", 1024),
        hv_bits=cfg.get("hv_bits", 4),
        top_k=cfg.get("top_k", 5),
        layers=cfg.get("layers"),
        max_len=cfg.get("max_len", 128),
        extract_batch_size=cfg.get("extract_batch_size", 8),
        vocab_size=cfg.get("vocab_size", 5000),
        reference_graphs=reference_graphs,
    )


__all__ = ["QwenAttentionGraphTMStudent", "load"]
