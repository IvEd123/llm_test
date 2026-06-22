"""Reusable inference pipeline: hidden-state matrix -> {class, confidence, evidence}.

Fits the final model on ALL 12 available samples (LOOCV already showed this
feature set is linearly separable -- see explore.py / models.py). Given n=12,
treat `important_features` as descriptive of THIS dataset's token-identity
signal, not a validated "math circuit".
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from analysis.load_data import load_samples, Sample
from analysis.features import pooled_stats, build_feature_matrix

LABELS = {1: "math", 0: "others"}


class MathPromptClassifier:
    def __init__(self, k_best: int = 30):
        self.k_best = k_best
        self.scaler = StandardScaler()
        self.selector = SelectKBest(f_classif, k=k_best)
        self.clf = LogisticRegression(max_iter=2000)
        self._fitted = False

    def fit(self, samples: list[Sample]) -> "MathPromptClassifier":
        X, y, _ = build_feature_matrix(samples, method="pooled")
        Xs = self.scaler.fit_transform(X)
        Xk = self.selector.fit_transform(Xs, y)
        self.clf.fit(Xk, y)
        self._fitted = True
        return self

    def predict(self, matrix: np.ndarray) -> dict:
        """matrix: (tokens, 2816) raw hidden-state activations for one prompt."""
        if not self._fitted:
            raise RuntimeError("call .fit(samples) first")

        feat = pooled_stats(matrix)[None, :]
        feat_s = self.scaler.transform(feat)
        feat_k = self.selector.transform(feat_s)

        proba = self.clf.predict_proba(feat_k)[0]
        pred = int(np.argmax(proba))

        selected_idx = self.selector.get_support(indices=True)
        coefs = self.clf.coef_[0]
        order = np.argsort(-np.abs(coefs))
        top_neurons = [
            {"neuron_idx": int(selected_idx[i] % 2816),  # mod 2816: pooled_stats stacks 5 stats blocks
             "pooled_stat_block": int(selected_idx[i] // 2816),  # 0=mean,1=max,2=min,3=std,4=median
             "weight": float(coefs[i])}
            for i in order[:10]
        ]

        return {
            "class": LABELS[pred],
            "confidence": float(proba[pred]),
            "evidence": {
                "important_features": [int(idx) for idx in selected_idx],
                "top_neurons": top_neurons,
            },
        }


def demo():
    samples = load_samples()
    model = MathPromptClassifier(k_best=30).fit(samples)

    print("Held-out-style spot check (model fit on all 12, predicting each back):")
    for s in samples:
        result = model.predict(s.matrix)
        mark = "OK" if result["class"] == s.label else "MISS"
        print(f"{s.name:10s} true={s.label:7s} pred={result['class']:7s} "
              f"conf={result['confidence']:.3f} [{mark}]")


if __name__ == "__main__":
    demo()
