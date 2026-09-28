import numpy as np
import pytest

from agro_rag.evaluation.metrics import (
    dedupe,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_matrix,
    score_rankings,
    summarize,
)

RANKINGS = [["a", "b", "c"], ["x", "y", "a"], ["x", "y", "z"], ["b", "d", "c"]]
RELEVANT = [{"a"}, {"a"}, {"a"}, {"c", "d"}]


def test_dedupe_keeps_first_occurrence_order() -> None:
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_relevance_matrix() -> None:
    matrix = relevance_matrix(RANKINGS, RELEVANT, depth=3)

    assert matrix.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1]]


def test_recall_hit_and_reciprocal_rank() -> None:
    matrix = relevance_matrix(RANKINGS, RELEVANT, depth=3)
    n_relevant = np.array([1, 1, 1, 2])

    assert recall_at_k(matrix, n_relevant, 1).tolist() == [1.0, 0.0, 0.0, 0.0]
    assert recall_at_k(matrix, n_relevant, 3).tolist() == [1.0, 1.0, 0.0, 1.0]
    assert recall_at_k(matrix, n_relevant, 2).tolist() == [1.0, 0.0, 0.0, 0.5]
    assert hit_rate_at_k(matrix, 2).tolist() == [1.0, 0.0, 0.0, 1.0]
    assert reciprocal_rank(matrix) == pytest.approx([1.0, 1 / 3, 0.0, 0.5])


def test_ndcg_penalizes_relevant_documents_that_were_not_retrieved() -> None:
    matrix = np.array([[1, 0], [0, 1], [0, 0]])

    scores = ndcg_at_k(matrix, np.array([1, 1, 1]), k=2)

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1 / np.log2(3))
    assert scores[2] == 0.0
    missing_one_of_two = ndcg_at_k(np.array([[1, 0]]), np.array([2]), k=2)
    assert missing_one_of_two[0] < 1.0


def test_score_rankings_and_summary() -> None:
    frame = score_rankings(RANKINGS, RELEVANT, ks=[1, 3])
    summary = summarize(frame)

    assert list(frame.columns) == [
        "reciprocal_rank",
        "recall@1",
        "hit@1",
        "ndcg@1",
        "recall@3",
        "hit@3",
        "ndcg@3",
    ]
    assert summary["recall@1"] == pytest.approx(0.25)
    assert summary["recall@3"] == pytest.approx(0.75)
    assert summary["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 0.5) / 4)
