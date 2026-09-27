import asyncio
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy.ext.asyncio import async_sessionmaker

from agro_rag.config import get_settings
from agro_rag.db.session import create_engine
from agro_rag.embeddings import get_embeddings
from agro_rag.ingestion.pipeline import IngestionStats, ingest_corpus

app = typer.Typer(help="agro-rag command line tools", no_args_is_help=True)


@app.callback()
def main() -> None:
    """agro-rag command line tools."""


async def _ingest(corpus: Path, collection: str, chunk_size: int, overlap: int) -> IngestionStats:
    engine = create_engine()
    try:
        async with async_sessionmaker(engine)() as session:
            return await ingest_corpus(
                session,
                get_embeddings(),
                corpus,
                collection=collection,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
    finally:
        await engine.dispose()


@app.command()
def ingest(
    corpus: Annotated[Path | None, typer.Option(help="Directory with markdown files")] = None,
    collection: Annotated[str | None, typer.Option()] = None,
    chunk_size: Annotated[int | None, typer.Option()] = None,
    overlap: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Chunk, embed and store the corpus in pgvector."""
    settings = get_settings()
    stats = asyncio.run(
        _ingest(
            corpus or settings.corpus_dir,
            collection or settings.collection,
            chunk_size or settings.chunk_size,
            overlap if overlap is not None else settings.chunk_overlap,
        )
    )
    typer.echo(
        f"documents={stats.documents} created={stats.created} updated={stats.updated} "
        f"skipped={stats.skipped} removed={stats.removed} chunks={stats.chunks}"
    )
