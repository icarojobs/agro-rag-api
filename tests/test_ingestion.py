from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from agro_rag.cli import app
from agro_rag.db.models import Chunk, Document
from agro_rag.embeddings import HashingEmbeddings
from agro_rag.ingestion.pipeline import ingest_corpus


async def _ingest(session: AsyncSession, corpus: Path) -> tuple[int, int, int, int]:
    stats = await ingest_corpus(
        session, HashingEmbeddings(), corpus, collection="t", chunk_size=60, chunk_overlap=10
    )
    return stats.created, stats.updated, stats.skipped, stats.removed


async def test_ingestion_is_idempotent_and_tracks_changes(
    session: AsyncSession, corpus_dir: Path
) -> None:
    assert await _ingest(session, corpus_dir) == (2, 0, 0, 0)
    chunks = await session.scalar(select(func.count()).select_from(Chunk))
    assert chunks is not None and chunks > 2

    assert await _ingest(session, corpus_dir) == (0, 0, 2, 0)

    (corpus_dir / "solo" / "calagem.md").write_text("# Calagem\n\nTexto novo.", encoding="utf-8")
    (corpus_dir / "pragas" / "percevejo.md").unlink()
    assert await _ingest(session, corpus_dir) == (0, 1, 0, 1)

    docs = (await session.scalars(select(Document))).all()
    assert [d.source for d in docs] == ["solo/calagem.md"]
    remaining = (await session.scalars(select(Chunk.content))).all()
    assert remaining == ["Calagem\nTexto novo."]


def test_cli_ingest(corpus_dir: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "--corpus",
            str(corpus_dir),
            "--collection",
            "cli",
            "--chunk-size",
            "80",
            "--overlap",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "documents=2 created=2" in result.output


async def test_changing_chunk_size_reindexes_documents(
    session: AsyncSession, corpus_dir: Path
) -> None:
    embeddings = HashingEmbeddings()
    first = await ingest_corpus(
        session, embeddings, corpus_dir, collection="t", chunk_size=60, chunk_overlap=10
    )
    second = await ingest_corpus(
        session, embeddings, corpus_dir, collection="t", chunk_size=400, chunk_overlap=10
    )

    assert (second.updated, second.skipped) == (2, 0)
    assert second.chunks < first.chunks
