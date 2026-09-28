import math
from typing import Any

import numpy as np
import pytest

from agro_rag import embeddings as emb
from agro_rag.config import get_settings
from agro_rag.db.models import EMBEDDING_DIM


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    model = emb.HashingEmbeddings()

    first, second = model.embed_documents(["calagem do solo", "calagem do solo"])

    assert first == second
    assert len(first) == EMBEDDING_DIM
    assert math.isclose(sum(v * v for v in first), 1.0, rel_tol=1e-9)


def test_hashing_embeddings_similar_texts_are_closer() -> None:
    model = emb.HashingEmbeddings()
    query = model.embed_query("calcário corrige acidez")
    related = model.embed_query("o calcário corrige a acidez do solo")
    unrelated = model.embed_query("percevejo ataca vagens de soja")

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, related) > dot(query, unrelated)


def test_empty_text_does_not_divide_by_zero() -> None:
    assert emb.HashingEmbeddings(dim=8).embed_query("") == [0.0] * 8


def test_sentence_transformer_wrapper_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, name: str, device: str) -> None:
            calls["init"] = (name, device)

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            calls["kwargs"] = kwargs
            return np.ones((len(texts), 3))

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    model = emb.SentenceTransformerEmbeddings("some/model")

    assert model.embed_query("x") == [1.0, 1.0, 1.0]
    assert calls["init"] == ("some/model", "cpu")
    assert calls["kwargs"]["normalize_embeddings"] is True


def test_get_embeddings_uses_configured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(emb.get_embeddings(), emb.HashingEmbeddings)

    emb.get_embeddings.cache_clear()
    get_settings.cache_clear()
    monkeypatch.setenv("AGRO_EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setattr(emb, "SentenceTransformerEmbeddings", lambda name: ("st", name))

    assert emb.get_embeddings() == ("st", get_settings().embedding_model)  # type: ignore[comparison-overlap]


def test_e5_models_get_query_and_passage_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class FakeModel:
        def __init__(self, name: str, device: str) -> None:
            pass

        def encode(self, texts: list[str], **_: Any) -> np.ndarray:
            seen.extend(texts)
            return np.zeros((len(texts), 2))

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    model = emb.SentenceTransformerEmbeddings("intfloat/multilingual-e5-small")

    model.embed_query("calagem")
    model.embed_documents(["texto"])

    assert seen == ["query: calagem", "passage: texto"]
