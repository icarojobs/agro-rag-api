import hashlib
import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

from agro_rag.config import get_settings
from agro_rag.db.models import EMBEDDING_DIM

if TYPE_CHECKING:
    pass

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# Models trained with asymmetric instructions expect different prefixes for
# queries and passages (e.g. the E5 family).
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
}


class SentenceTransformerEmbeddings(Embeddings):
    """Local Hugging Face model; vectors are L2-normalized so cosine == dot product."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix, self.document_prefix = KNOWN_PREFIXES.get(model_name, ("", ""))
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=32, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings via feature hashing. Used in tests and CI."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.embedding_provider == "hashing":
        return HashingEmbeddings()
    return SentenceTransformerEmbeddings(settings.embedding_model)
