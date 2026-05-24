"""Tsetlin-family students plugged into the paper-c robustness harness.

Each module in this package exposes a ``load(checkpoint_dir, dataset, ...)``
function returning an object that implements the contract documented in
``GraphTMStudent``: a ``predict_logits(texts) -> ndarray[(n, n_classes)]``
method, plus the attributes used by ``TMModelWrapper``.

The dispatcher ``load_student(family, ...)`` selects between the three
implementations.
"""

from .base import GraphTMStudent


def load_student(family, checkpoint_dir, dataset=None, **kwargs):
    """Load a student by family name.

    Args:
        family: one of "subword_dep", "bert_attention", "qwen_attention".
        checkpoint_dir: directory containing the saved student. Expected
            contents are documented in each module's ``load`` docstring.
        dataset: dataset tag passed to the loader for vocab and label
            resolution (R8, R52, IMDb).
        **kwargs: forwarded to the family-specific ``load`` function.

    Returns:
        A ``GraphTMStudent`` instance.
    """
    family = family.lower()
    if family in ("subword_dep", "subword-dep", "paper_a"):
        from . import subword_dep_graphtm
        return subword_dep_graphtm.load(checkpoint_dir, dataset=dataset, **kwargs)
    if family in ("bert_attention", "bert-attention", "paper_b"):
        from . import bert_attention_graphtm
        return bert_attention_graphtm.load(checkpoint_dir, dataset=dataset, **kwargs)
    if family in ("qwen_attention", "qwen-attention", "decoder"):
        from . import qwen_attention_graphtm
        return qwen_attention_graphtm.load(checkpoint_dir, dataset=dataset, **kwargs)
    raise ValueError(
        f"Unknown student family: {family}. "
        "Choose from: subword_dep, bert_attention, qwen_attention."
    )


__all__ = ["GraphTMStudent", "load_student"]
