"""API contract for Tsetlin-family students consumed by TMModelWrapper.

A student wraps a trained Tsetlin-Machine variant so that the paper-c
robustness harness can drive it through TextAttack. The original
``TMModelWrapper`` in ``robustness/tm_wrapper.py`` was written for the
bag-of-words drop-clause TM. GraphTM-based students need different
inputs (per-document graph reconstruction), but they expose the same
external behaviour: take a list of strings, return a logits ndarray.

The contract is intentionally narrow:

    predict_logits(texts: list[str]) -> ndarray of shape (n, n_classes)
        Real-valued scores, one row per input. Higher means the model
        favours that class. Callers may treat the matrix as logits or
        normalise it; the harness does not assume softmax-normalised
        probabilities.

    n_classes: int
        Number of output classes. TextAttack uses this for budgeting.

    family: str
        Short identifier used in log filenames and reports.

    __call__(texts) -> ndarray
        Convenience alias for predict_logits, matching the existing
        TMModelWrapper.__call__ signature so the same attack_runner
        functions can drive it without modification.

Differentiability: Tsetlin Machines are not differentiable. A student
must raise ``NotImplementedError`` from any gradient-related call. The
``backward()`` stub below is provided so that TextAttack hooks that
introspect for it get a clear error rather than a silent zero gradient.
"""

from __future__ import annotations

from typing import List

import numpy as np


class GraphTMStudent:
    """Base class locking the predict_logits contract.

    Subclasses must implement ``predict_logits`` and set ``n_classes``,
    ``family``, and ``label_names``.
    """

    family: str = "abstract"
    n_classes: int = 0
    label_names: List[str] = []

    def predict_logits(self, texts):
        """Return an (n, n_classes) ndarray of real-valued scores.

        Args:
            texts: list of strings (may also be a single string).

        Returns:
            ndarray of shape (n, n_classes), dtype float32.
        """
        raise NotImplementedError

    def __call__(self, text_input_list):
        """TextAttack-compatible entry point.

        Accepts either a string or a list of strings. Always returns
        an ndarray of shape (n, n_classes).
        """
        if isinstance(text_input_list, str):
            text_input_list = [text_input_list]
        logits = self.predict_logits(list(text_input_list))
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        if logits.shape[1] != self.n_classes:
            raise ValueError(
                f"{self.family}: predict_logits returned shape {logits.shape}; "
                f"expected (n, {self.n_classes})."
            )
        return logits

    def backward(self, *args, **kwargs):
        """TMs are non-differentiable. Calling backward is a programming
        error: raise to surface it immediately.
        """
        raise NotImplementedError(
            f"{self.family}: GraphTM students are not differentiable; "
            "gradient-based attacks cannot be applied directly. Use a "
            "decision-based recipe such as TextFooler or BERT-Attack."
        )

    def _scores_to_logits(self, raw_scores):
        """Normalise a (n, n_classes) score matrix to float32 ndarray.

        Centralised so the three concrete students share the same
        post-processing. The GraphTM ``.score()`` method returns class
        vote sums in int32; we just cast to float32 here. We do not
        softmax: TextAttack does not require normalised probabilities,
        and softmax would suppress vote-margin information that
        per-sample logs use downstream.
        """
        arr = np.asarray(raw_scores, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.n_classes:
            raise ValueError(
                f"{self.family}: raw score matrix has shape {arr.shape}; "
                f"expected (n, {self.n_classes})."
            )
        return arr


__all__ = ["GraphTMStudent"]
