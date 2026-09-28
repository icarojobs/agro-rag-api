import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
from langchain_core.embeddings import Embeddings
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

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


async def ingest_documents(
    session: AsyncSession,
    embeddings: Embeddings,
    docs: list[SourceDocument],
    *,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestionStats:
    stats = IngestionStats(documents=len(docs))
    existing = {
        d.source: d
        for d in (
            await session.scalars(select(Document).where(Document.collection == collection))
        ).all()
    }

    for doc in docs:
        current = existing.pop(doc.source, None)
        if current is not None and current.content_hash == doc.content_hash:
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
            content_hash=doc.content_hash,
            chunks=[
                Chunk(chunk_index=i, content=text, embedding=vec)
                for i, (text, vec) in enumerate(zip(pieces, vectors, strict=True))
            ],
        )
        session.add(record)
        stats.chunks += len(pieces)

    if existing:
        await session.execute(
            delete(Document).where(Document.id.in_(d.id for d in existing.values()))
        )
        stats.removed = len(existing)

    await session.commit()
    logger.info("ingestion_finished", collection=collection, **asdict(stats))
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
