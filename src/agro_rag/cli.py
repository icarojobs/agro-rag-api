import asyncio
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from sqlalchemy.ext.asyncio import async_sessionmaker

from agro_rag.aws import aws_enabled, ensure_infrastructure
from agro_rag.cache import close_cache
from agro_rag.config import get_settings
from agro_rag.db.session import create_engine
from agro_rag.embeddings import get_embeddings
from agro_rag.evaluation.runner import (
    EvalConfig,
    EvalResult,
    evaluate_config,
    load_questions,
    log_to_mlflow,
)
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
        await close_cache()


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


@app.command("aws-init")
def aws_init() -> None:
    """Create the SNS topic, SQS queues (with DLQ) and DynamoDB table used by ingestion."""
    settings = get_settings()
    if not aws_enabled(settings):
        typer.echo("AGRO_AWS_ENDPOINT_URL is not set", err=True)
        raise typer.Exit(1)
    infra = asyncio.run(ensure_infrastructure(settings))
    typer.echo(f"topic={infra.topic_arn} queue={infra.queue_url} dlq={infra.dlq_url}")
    typer.echo(f"audit={infra.audit_queue_url} table={infra.table}")


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


async def _evaluate(
    questions_path: Path,
    corpus: Path,
    configs: list[EvalConfig],
    tracking_uri: str | None,
    experiment: str,
) -> list[EvalResult]:
    questions = load_questions(questions_path)
    embeddings = get_embeddings()
    engine = create_engine()
    results = []
    try:
        async with async_sessionmaker(engine)() as session:
            for config in configs:
                result = await evaluate_config(session, embeddings, corpus, questions, config)
                if tracking_uri:
                    result.params["mlflow_run_id"] = log_to_mlflow(result, tracking_uri, experiment)
                results.append(result)
    finally:
        await engine.dispose()
    return results


@app.command()
def evaluate(
    questions: Annotated[Path, typer.Option(help="JSONL with id, question, relevant")] = Path(
        "eval/questions.jsonl"
    ),
    corpus: Annotated[Path | None, typer.Option()] = None,
    chunk_sizes: Annotated[str, typer.Option(help="Comma-separated chunk sizes")] = "300,500,800",
    overlaps: Annotated[str, typer.Option(help="Comma-separated overlaps")] = "50",
    k: Annotated[str, typer.Option(help="Cut-offs for recall/nDCG")] = "1,3,5,10",
    mlflow_uri: Annotated[str | None, typer.Option(help="Empty string disables MLflow")] = None,
    experiment: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Measure retrieval quality (recall@k, MRR, nDCG) and log each run to MLflow."""
    settings = get_settings()
    model = settings.embedding_model if settings.embedding_provider != "hashing" else "hashing"
    configs = [
        EvalConfig(
            chunk_size=size, chunk_overlap=overlap, ks=tuple(_int_list(k)), embedding_model=model
        )
        for size in _int_list(chunk_sizes)
        for overlap in _int_list(overlaps)
    ]
    tracking_uri = settings.mlflow_tracking_uri if mlflow_uri is None else mlflow_uri
    results = asyncio.run(
        _evaluate(
            questions,
            corpus or settings.corpus_dir,
            configs,
            tracking_uri or None,
            experiment or settings.mlflow_experiment,
        )
    )
    table = pd.DataFrame(
        [
            {
                "chunk_size": r.config.chunk_size,
                "overlap": r.config.chunk_overlap,
                "chunks": r.params["n_chunks"],
                **r.summary,
            }
            for r in results
        ]
    )
    typer.echo(table.round(3).to_string(index=False))
