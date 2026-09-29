import asyncio
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
from langchain_core.embeddings import Embeddings
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.cache import invalidate_collection
from agro_rag.db.models import Chunk, Document
from agro_rag.ingestion.chunking import chunk_document
from agro_rag.ingestion.loader import SourceDocument, load_corpus

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class IngestionStats:
    documents: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0


def _fingerprint(
    doc: SourceDocument, embeddings: Embeddings, chunk_size: int, chunk_overlap: int
) -> str:
    """Changes whenever the text, the embedding model or the chunking parameters change."""
    model = getattr(embeddings, "model_name", type(embeddings).__name__)
    key = f"{doc.content_hash}:{model}:{chunk_size}:{chunk_overlap}"
    return hashlib.sha256(key.encode()).hexdigest()


async def ingest_documents(
    session: AsyncSession,
    embeddings: Embeddings,
    docs: list[SourceDocument],
    *,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    prune: bool = True,
) -> IngestionStats:
    """Index `docs`. With `prune`, documents of the collection missing from `docs` are removed."""
    stats = IngestionStats(documents=len(docs))
    existing = {
        d.source: d
        for d in (
            await session.scalars(select(Document).where(Document.collection == collection))
        ).all()
    }

    for doc in docs:
        current = existing.pop(doc.source, None)
        fingerprint = _fingerprint(doc, embeddings, chunk_size, chunk_overlap)
        if current is not None and current.content_hash == fingerprint:
            stats.skipped += 1
            continue
        if current is not None:
            await session.delete(current)
            await session.flush()
            stats.updated += 1
        else:
            stats.created += 1

        pieces = chunk_document(doc, chunk_size, chunk_overlap)
        vectors = await asyncio.to_thread(embeddings.embed_documents, pieces)
        record = Document(
            collection=collection,
            source=doc.source,
            title=doc.title,
            category=doc.category,
            content_hash=fingerprint,
            chunks=[
                Chunk(chunk_index=i, content=text, embedding=vec)
                for i, (text, vec) in enumerate(zip(pieces, vectors, strict=True))
            ],
        )
        session.add(record)
        stats.chunks += len(pieces)

    if prune and existing:
        await session.execute(
            delete(Document).where(Document.id.in_(d.id for d in existing.values()))
        )
        stats.removed = len(existing)

    await session.commit()
    logger.info("ingestion_finished", collection=collection, **asdict(stats))
    if stats.created or stats.updated or stats.removed:
        await invalidate_collection(collection)
    return stats


async def ingest_corpus(
    session: AsyncSession,
    embeddings: Embeddings,
    corpus_dir: Path,
    *,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestionStats:
    return await ingest_documents(
        session,
        embeddings,
        load_corpus(corpus_dir),
        collection=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
