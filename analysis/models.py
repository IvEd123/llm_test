"""Binary classifiers for math vs others, evaluated with leave-one-out CV.

n=12 total -- far too small for a held-out test split to mean anything, so
LOOCV (12 folds, each leaving exactly one sample out) is used throughout.
Results here demonstrate the pipeline mechanics; they are NOT a reliable
estimate of real-world generalization given the sample size.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from analysis.load_data import load_samples
from analysis.features import build_feature_matrix

MODELS = {
    "logreg": LogisticRegression(max_iter=2000, C=1.0),
    "svm_linear": SVC(kernel="linear", probability=True),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "mlp": MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000, random_state=0),
}


def make_pipeline(model, k_best: int = 30) -> Pipeline:
    """Scale -> univariate feature selection (critical: width >> n_samples) -> classifier."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_best)),
        ("clf", model),
    ])


def loocv_eval(X: np.ndarray, y: np.ndarray, model, k_best: int = 30) -> dict:
    loo = LeaveOneOut()
    y_true, y_pred, y_score = [], [], []
    for train_idx, test_idx in loo.split(X):
        pipe = make_pipeline(model, k_best=k_best)
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        y_true.append(y[test_idx][0])
        y_pred.append(pred[0])
        if hasattr(pipe, "predict_proba"):
            y_score.append(pipe.predict_proba(X[test_idx])[0, 1])
        else:
            y_score.append(pred[0])

    y_true, y_pred, y_score = map(np.array, (y_true, y_pred, y_score))
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")  # only one class present in scores
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def run_all(feature_method: str = "pooled", k_best: int = 30) -> dict:
    samples = load_samples()
    X, y, names = build_feature_matrix(samples, method=feature_method)

    results = {}
    for name, model in MODELS.items():
        results[name] = loocv_eval(X, y, model, k_best=k_best)
    return results


if __name__ == "__main__":
    for method in ["pooled", "last_token", "first_token"]:
        print(f"\n=== feature method: {method} ===")
        res = run_all(feature_method=method, k_best=30)
        for name, r in res.items():
            print(f"{name:15s} acc={r['accuracy']:.2f} prec={r['precision']:.2f} "
                  f"rec={r['recall']:.2f} f1={r['f1']:.2f} auc={r['roc_auc']:.2f}")
