"""Deterministic smoke-test backends; these are not learned semantic models."""

import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class OfflineEncoder:
    def __init__(self):
        self.vectorizer = HashingVectorizer(
            n_features=1024, alternate_sign=False, ngram_range=(1, 2), norm="l2"
        )

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        matrix = self.vectorizer.transform([texts] if single else texts).toarray()
        return matrix[0] if single else matrix


class OfflineReranker:
    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q = set(re.findall(r"\w+", query.lower()))
            d = set(re.findall(r"\w+", text.lower()))
            scores.append(len(q & d) / max(1, len(q)))
        return np.asarray(scores)
