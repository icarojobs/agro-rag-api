from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.db.models import EMBEDDING_DIM, Chunk, Document


async def test_document_with_chunks_roundtrip(session: AsyncSession) -> None:
    doc = Document(
        collection="default",
        source="solo/calagem.md",
        title="Calagem",
        category="solo",
        content_hash="abc",
    )
    doc.chunks.append(Chunk(chunk_index=0, content="texto", embedding=[0.1] * EMBEDDING_DIM))
    session.add(doc)
    await session.commit()

    stored = (await session.execute(select(Chunk))).scalar_one()
    assert stored.document_id == doc.id
    assert len(stored.embedding) == EMBEDDING_DIM
