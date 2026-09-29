"""SQS consumer that indexes the documents queued by POST /ingest/documents.

Delivery is at-least-once, so processing is idempotent twice over: a job already marked
`done` in DynamoDB is acknowledged without work, and the pipeline itself skips a document
whose content hash is unchanged. A message is deleted only after the job succeeded; on any
failure it becomes visible again after the visibility timeout and, after
`ingestion_max_receive_count` receives, SQS moves it to the dead-letter queue.
"""

import asyncio
from typing import Any, Literal

import structlog
from langchain_core.embeddings import Embeddings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agro_rag.aws import Infra, aws_client, ensure_infrastructure
from agro_rag.config import Settings
from agro_rag.ingestion.jobs import IngestionMessage, JobAlreadyDoneError, JobStore
from agro_rag.ingestion.loader import SourceDocument
from agro_rag.ingestion.pipeline import ingest_documents

logger = structlog.get_logger(__name__)

Outcome = Literal["processed", "duplicate"]


async def process_message(
    body: str,
    *,
    store: JobStore,
    sessionmaker: async_sessionmaker[AsyncSession],
    embeddings: Embeddings,
    settings: Settings,
) -> Outcome:
    """Index one queued document. Raises on failure so the message is not acknowledged."""
    message = IngestionMessage.model_validate_json(body)
    try:
        await store.start(message.job_id)
    except JobAlreadyDoneError:
        logger.info("ingestion_duplicate", job_id=message.job_id)
        return "duplicate"
    try:
        async with sessionmaker() as session:
            stats = await ingest_documents(
                session,
                embeddings,
                [
                    SourceDocument(
                        source=message.source,
                        title=message.title,
                        category=message.category,
                        body=message.content,
                    )
                ],
                collection=message.collection,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                prune=False,
            )
    except Exception as exc:
        await store.fail(message.job_id, f"{type(exc).__name__}: {exc}")
        raise
    await store.complete(
        message.job_id,
        {
            "created": stats.created,
            "updated": stats.updated,
            "skipped": stats.skipped,
            "chunks": stats.chunks,
        },
    )
    return "processed"


async def _handle(
    sqs: Any,
    infra: Infra,
    raw: dict[str, Any],
    *,
    store: JobStore,
    sessionmaker: async_sessionmaker[AsyncSession],
    embeddings: Embeddings,
    settings: Settings,
) -> None:
    receives = raw.get("Attributes", {}).get("ApproximateReceiveCount")
    try:
        outcome = await process_message(
            raw["Body"],
            store=store,
            sessionmaker=sessionmaker,
            embeddings=embeddings,
            settings=settings,
        )
    except Exception as exc:
        # Not deleted: it returns after the visibility timeout, then goes to the DLQ.
        logger.warning(
            "ingestion_failed", error=type(exc).__name__, detail=str(exc)[:200], receives=receives
        )
        return
    await sqs.delete_message(QueueUrl=infra.queue_url, ReceiptHandle=raw["ReceiptHandle"])
    logger.info("ingestion_acknowledged", outcome=outcome)


async def run_worker(
    settings: Settings,
    stop: asyncio.Event,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    embeddings: Embeddings,
) -> None:
    infra = await ensure_infrastructure(settings)
    store = JobStore(settings, infra)
    logger.info("ingestion_worker_started", queue=infra.queue_url)
    async with aws_client(settings, "sqs") as sqs:
        while not stop.is_set():
            poll = asyncio.ensure_future(
                sqs.receive_message(
                    QueueUrl=infra.queue_url,
                    MaxNumberOfMessages=settings.ingestion_batch_size,
                    WaitTimeSeconds=settings.ingestion_poll_wait_seconds,
                    AttributeNames=["ApproximateReceiveCount"],
                )
            )
            waiter = asyncio.ensure_future(stop.wait())
            await asyncio.wait({poll, waiter}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if not poll.done():
                poll.cancel()
                break
            for raw in poll.result().get("Messages", []):
                await _handle(
                    sqs,
                    infra,
                    raw,
                    store=store,
                    sessionmaker=sessionmaker,
                    embeddings=embeddings,
                    settings=settings,
                )
    logger.info("ingestion_worker_stopped")
