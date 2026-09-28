from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import ndcg_score

IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


def dedupe(sources: Iterable[str]) -> list[str]:
    """Collapse chunk-level results into a ranked list of unique documents."""
    return list(dict.fromkeys(sources))


def relevance_matrix(
    rankings: Sequence[Sequence[str]], relevant: Sequence[set[str]], depth: int
) -> IntArray:
    matrix = np.zeros((len(rankings), depth), dtype=int)
    for i, (ranked, gold) in enumerate(zip(rankings, relevant, strict=True)):
        for j, source in enumerate(ranked[:depth]):
            matrix[i, j] = int(source in gold)
    return matrix


def recall_at_k(matrix: IntArray, n_relevant: IntArray, k: int) -> FloatArray:
    return matrix[:, :k].sum(axis=1) / n_relevant


def hit_rate_at_k(matrix: IntArray, k: int) -> FloatArray:
    return (matrix[:, :k].sum(axis=1) > 0).astype(float)


def reciprocal_rank(matrix: IntArray) -> FloatArray:
    hits = matrix.argmax(axis=1)
    found = matrix.max(axis=1) > 0
    return np.where(found, 1.0 / (hits + 1), 0.0)


def ndcg_at_k(matrix: IntArray, n_relevant: IntArray, k: int) -> FloatArray:
    """Per-query nDCG@k with sklearn.

    Relevant documents that were never retrieved are appended with the lowest
    score, so the ideal DCG accounts for them.
    """
    depth = matrix.shape[1]
    scores = []
    for row, total in zip(matrix, n_relevant, strict=True):
        missing = max(int(total) - int(row.sum()), 0)
        y_true = np.concatenate([row, np.ones(missing, dtype=int)])
        y_score = np.concatenate([np.arange(depth, 0, -1), np.zeros(missing)])
        if y_true.sum() == 0:
            scores.append(0.0)
            continue
        scores.append(float(ndcg_score([y_true], [y_score], k=k)))
    return np.asarray(scores)


def score_rankings(
    rankings: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
    ks: Sequence[int],
) -> pd.DataFrame:
    """One row per question with recall@k, hit@k, nDCG@k and reciprocal rank."""
    depth = max(ks)
    matrix = relevance_matrix(rankings, relevant, depth)
    n_relevant = np.array([len(r) for r in relevant])
    frame = pd.DataFrame({"reciprocal_rank": reciprocal_rank(matrix)})
    for k in ks:
        frame[f"recall@{k}"] = recall_at_k(matrix, n_relevant, k)
        frame[f"hit@{k}"] = hit_rate_at_k(matrix, k)
        frame[f"ndcg@{k}"] = ndcg_at_k(matrix, n_relevant, k)
    return frame


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    summary = {col: float(frame[col].mean()) for col in frame.columns if "@" in col}
    summary["mrr"] = float(frame["reciprocal_rank"].mean())
    return summary
