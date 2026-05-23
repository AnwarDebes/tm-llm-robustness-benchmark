"""TextAttack-compatible wrapper for Tsetlin Machine classifiers.

Wraps TMClassifier (or any model with .predict()) to work with TextAttack's
adversarial attack framework. Supports both binary and multi-class.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2


class TMModelWrapper:
    """TextAttack-compatible wrapper for a trained TM classifier.

    TextAttack expects:
    - __call__(text_input_list) -> list of dicts or np.array of logits/probs

    Since TMs don't produce probabilities, we return clause-vote-based
    pseudo-probabilities (normalized vote sums per class).
    """

    def __init__(self, tm_model, vectorizer, feature_selector=None, n_classes=2):
        """
        Args:
            tm_model: trained TMClassifier or TMCoalescedClassifier
            vectorizer: fitted CountVectorizer
            feature_selector: fitted SelectKBest (optional)
            n_classes: number of output classes
        """
        self.model = tm_model
        self.vectorizer = vectorizer
        self.selector = feature_selector
        self.n_classes = n_classes

    def _texts_to_features(self, text_list):
        """Convert text strings to feature matrix."""
        X = self.vectorizer.transform(text_list)
        if self.selector is not None:
            X = self.selector.transform(X)
        return X.toarray().astype(np.uint32)

    def __call__(self, text_input_list):
        """Predict on a list of text strings.

        Returns np.array of shape (batch, n_classes) with pseudo-probabilities.
        """
        if isinstance(text_input_list, str):
            text_input_list = [text_input_list]

        X = self._texts_to_features(text_input_list)
        preds = self.model.predict(X)

        # Convert hard predictions to one-hot pseudo-probabilities.
        # TMs do not expose class scores natively.
        probs = np.zeros((len(preds), self.n_classes), dtype=np.float32)
        for i, p in enumerate(preds):
            probs[i, int(p)] = 1.0

        return probs


def build_tm_for_attack(
    train_texts, train_labels, test_texts, test_labels,
    num_clauses=10000, T=8000, s=2.0,
    features=5000, max_ngram=2,
    weighted_clauses=True, clause_drop_p=0.75,
    platform="CUDA", seed=42, epochs=20,
):
    """Train a TM and return it wrapped for TextAttack.

    Returns:
        (wrapper, clean_accuracy)
    """
    from tmu.models.classification.vanilla_classifier import TMClassifier

    n_classes = len(set(train_labels))

    # Vectorize
    vectorizer = CountVectorizer(
        ngram_range=(1, max_ngram), binary=True, max_features=50000
    )
    X_train_raw = vectorizer.fit_transform(train_texts)
    X_test_raw = vectorizer.transform(test_texts)

    # Feature selection
    skb = SelectKBest(chi2, k=features)
    skb.fit(X_train_raw, np.array(train_labels))
    X_train = skb.transform(X_train_raw).toarray().astype(np.uint32)
    X_test = skb.transform(X_test_raw).toarray().astype(np.uint32)
    Y_train = np.array(train_labels, dtype=np.uint32)
    Y_test = np.array(test_labels, dtype=np.uint32)

    # Train
    tm = TMClassifier(
        num_clauses, T, s,
        platform=platform,
        weighted_clauses=weighted_clauses,
        clause_drop_p=clause_drop_p,
        seed=seed,
    )

    best_acc = 0.0
    for epoch in range(epochs):
        tm.fit(X_train, Y_train, shuffle=True)
        preds = tm.predict(X_test)
        acc = (preds == Y_test).mean()
        if acc > best_acc:
            best_acc = acc

    wrapper = TMModelWrapper(tm, vectorizer, skb, n_classes=n_classes)
    return wrapper, float(best_acc)
