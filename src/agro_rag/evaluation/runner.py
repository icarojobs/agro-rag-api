import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from langchain_core.embeddings import Embeddings
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.db.models import Chunk, Document
from agro_rag.evaluation.metrics import dedupe, score_rankings, summarize
from agro_rag.ingestion.pipeline import ingest_corpus
from agro_rag.retrieval import search


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    relevant: frozenset[str]


@dataclass(frozen=True, slots=True)
class EvalConfig:
    chunk_size: int
    chunk_overlap: int
    ks: tuple[int, ...] = (1, 3, 5, 10)
    embedding_model: str = "hashing"

    @property
    def collection(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.embedding_model.split("/")[-1].lower())
        return f"eval-{slug}-{self.chunk_size}-{self.chunk_overlap}"[:64]


@dataclass(slots=True)
class EvalResult:
    config: EvalConfig
    summary: dict[str, float]
    per_question: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            questions.append(Question(raw["id"], raw["question"], frozenset(raw["relevant"])))
    return questions


async def evaluate_config(
    session: AsyncSession,
    embeddings: Embeddings,
    corpus_dir: Path,
    questions: list[Question],
    config: EvalConfig,
) -> EvalResult:
    stats = await ingest_corpus(
        session,
        embeddings,
        corpus_dir,
        collection=config.collection,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    depth = max(config.ks)
    rankings: list[list[str]] = []
    latencies_ms: list[float] = []
    for q in questions:
        started = time.perf_counter()
        # Over-fetch chunks so that `depth` unique documents survive deduplication.
        chunks = await search(
            session, embeddings, q.question, k=depth * 3, collection=config.collection
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        rankings.append(dedupe(c.source for c in chunks)[:depth])

    per_question = score_rankings(rankings, [set(q.relevant) for q in questions], config.ks)
    per_question.insert(0, "id", [q.id for q in questions])
    per_question.insert(1, "question", [q.question for q in questions])
    per_question["top_sources"] = [" | ".join(r[:3]) for r in rankings]
    per_question["latency_ms"] = latencies_ms

    summary = summarize(per_question.drop(columns=["id", "question", "top_sources"]))
    summary["latency_p50_ms"] = float(np.percentile(latencies_ms, 50))
    summary["latency_p95_ms"] = float(np.percentile(latencies_ms, 95))

    total_chunks = stats.chunks or await _count_chunks(session, config.collection)
    params = {
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "ks": ",".join(map(str, config.ks)),
        "n_questions": len(questions),
        "n_documents": stats.documents,
        "n_chunks": total_chunks,
    }
    return EvalResult(config=config, summary=summary, per_question=per_question, params=params)


async def _count_chunks(session: AsyncSession, collection: str) -> int:
    stmt = (
        select(func.count())
        .select_from(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.collection == collection)
    )
    return int(await session.scalar(stmt) or 0)


def log_to_mlflow(result: EvalResult, tracking_uri: str, experiment: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    run_name = f"{result.config.collection}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(result.params)
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in result.summary.items()})
        mlflow.log_text(result.per_question.to_csv(index=False), "per_question.csv")
        mlflow.set_tag("stage", "retrieval")
        return str(run.info.run_id)
