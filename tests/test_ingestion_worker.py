import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from agro_rag.aws import Infra, aws_client
from agro_rag.cli import app
from agro_rag.config import Settings, get_settings
from agro_rag.db.session import get_sessionmaker
from agro_rag.embeddings import HashingEmbeddings
from agro_rag.ingestion.jobs import IngestionMessage, JobStore
from agro_rag.ingestion.queue import IngestionQueue
from agro_rag.ingestion.worker import process_message, run_worker
from agro_rag.retrieval import search

pytestmark = pytest.mark.skipif(
    not os.environ.get("AGRO_AWS_ENDPOINT_URL"), reason="needs an AWS emulator (floci)"
)

DOC = {
    "source": "solo/novo.md",
    "title": "Calagem em plantio direto",
    "category": "solo",
    "content": "A calagem em plantio direto é feita na superfície, sem incorporação.",
}


class CountingEmbeddings(HashingEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return super().embed_documents(texts)


class BrokenEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model unavailable")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("model unavailable")


def _body(job_id: str, **overrides: Any) -> str:
    return IngestionMessage(
        job_id=job_id, collection="default", **{**DOC, **overrides}
    ).model_dump_json()


@pytest.fixture
def store(isolated: Infra) -> JobStore:
    return JobStore(get_settings(), isolated)


async def _queued(store: JobStore, job_id: str) -> str:
    body = _body(job_id)
    await store.create(job_id, IngestionMessage.model_validate_json(body))
    return body


async def _process(body: str, store: JobStore, embeddings: Embeddings) -> str:
    return await process_message(
        body,
        store=store,
        sessionmaker=get_sessionmaker(),
        embeddings=embeddings,
        settings=get_settings(),
    )


async def test_process_message_indexes_the_document_and_completes_the_job(
    store: JobStore, session: AsyncSession
) -> None:
    body = await _queued(store, "j-ok")

    assert await _process(body, store, HashingEmbeddings()) == "processed"

    job = await store.get("j-ok")
    assert job is not None
    assert (job["status"], job["attempts"]) == ("done", 1)
    assert job["result"]["created"] == 1
    hits = await search(
        session, HashingEmbeddings(), "calagem plantio direto", k=1, collection="default"
    )
    assert hits[0].source == "solo/novo.md"


async def test_process_message_does_not_prune_other_documents(
    store: JobStore, session: AsyncSession
) -> None:
    await _process(await _queued(store, "j-1"), store, HashingEmbeddings())
    other = _body("j-2", source="pragas/outro.md", title="Outro", content="Texto sobre pragas.")
    await store.create("j-2", IngestionMessage.model_validate_json(other))
    await _process(other, store, HashingEmbeddings())

    hits = await search(session, HashingEmbeddings(), "calagem pragas", k=5, collection="default")

    assert {h.source for h in hits} == {"solo/novo.md", "pragas/outro.md"}


async def test_redelivered_message_is_acknowledged_without_reprocessing(store: JobStore) -> None:
    embeddings = CountingEmbeddings()
    body = await _queued(store, "j-dup")

    assert await _process(body, store, embeddings) == "processed"
    assert await _process(body, store, embeddings) == "duplicate"

    assert embeddings.document_calls == 1
    job = await store.get("j-dup")
    assert job is not None
    assert job["attempts"] == 1


async def test_failure_marks_the_job_failed_and_a_retry_can_succeed(store: JobStore) -> None:
    body = await _queued(store, "j-retry")

    with pytest.raises(RuntimeError, match="model unavailable"):
        await _process(body, store, BrokenEmbeddings())
    failed = await store.get("j-retry")
    assert await _process(body, store, HashingEmbeddings()) == "processed"
    done = await store.get("j-retry")

    assert failed is not None
    assert failed["status"] == "failed"
    assert "model unavailable" in failed["error"]
    assert done is not None
    assert (done["status"], done["attempts"]) == ("done", 2)


async def test_poison_message_raises(store: JobStore) -> None:
    with pytest.raises(ValueError, match="valid"):
        await _process("not json at all", store, HashingEmbeddings())


@pytest.fixture
async def worker(isolated: Infra) -> AsyncIterator[dict[str, Any]]:
    """Runs the real consumer loop against floci until the test ends."""
    holder: dict[str, Any] = {"embeddings": HashingEmbeddings()}

    async def start() -> None:
        holder["stop"] = asyncio.Event()
        holder["task"] = asyncio.create_task(
            run_worker(
                get_settings(),
                holder["stop"],
                sessionmaker=get_sessionmaker(),
                embeddings=holder["embeddings"],
            )
        )

    holder["start"] = start
    yield holder
    if "stop" in holder:
        holder["stop"].set()
        await asyncio.wait_for(holder["task"], timeout=15)


async def _until(predicate: Any, within: float = 20) -> Any:
    async with asyncio.timeout(within):
        while not (value := await predicate()):  # noqa: ASYNC110
            await asyncio.sleep(0.3)
    return value


async def test_worker_consumes_a_queued_job_end_to_end(
    worker: dict[str, Any], isolated: Infra, session: AsyncSession
) -> None:
    await worker["start"]()
    job_id, _ = await IngestionQueue(get_settings()).enqueue(
        IngestionMessage(job_id="", collection="default", **DOC)
    )
    store = JobStore(get_settings(), isolated)

    async def done() -> Any:
        job = await store.get(job_id)
        return job if job and job["status"] == "done" else None

    job = await _until(done)

    assert job["attempts"] == 1
    hits = await search(session, HashingEmbeddings(), "calagem plantio", k=1, collection="default")
    assert hits[0].source == "solo/novo.md"


async def _dlq_bodies(settings: Settings, infra: Infra) -> list[str]:
    async with aws_client(settings, "sqs") as sqs:
        response = await sqs.receive_message(
            QueueUrl=infra.dlq_url, WaitTimeSeconds=1, MaxNumberOfMessages=10
        )
    return [m["Body"] for m in response.get("Messages", [])]


async def test_poison_message_ends_in_the_dlq_after_max_receives(
    worker: dict[str, Any], isolated: Infra
) -> None:
    await worker["start"]()
    async with aws_client(get_settings(), "sns") as sns:
        await sns.publish(TopicArn=isolated.topic_arn, Message="{definitely not a job")

    async def in_dlq() -> Any:
        return await _dlq_bodies(get_settings(), isolated) or None

    assert await _until(in_dlq, within=30) == ["{definitely not a job"]


async def test_failing_job_is_retried_then_dead_lettered(
    worker: dict[str, Any], isolated: Infra
) -> None:
    worker["embeddings"] = BrokenEmbeddings()
    await worker["start"]()
    job_id, _ = await IngestionQueue(get_settings()).enqueue(
        IngestionMessage(job_id="", collection="default", **DOC)
    )

    async def in_dlq() -> Any:
        return await _dlq_bodies(get_settings(), isolated) or None

    (body,) = await _until(in_dlq, within=30)

    job = await JobStore(get_settings(), isolated).get(job_id)
    assert json.loads(body)["job_id"] == job_id
    assert job is not None
    assert (job["status"], job["attempts"]) == ("failed", 2)


async def test_worker_stops_promptly_on_signal(worker: dict[str, Any]) -> None:
    await worker["start"]()
    await asyncio.sleep(1)

    worker["stop"].set()
    await asyncio.wait_for(worker["task"], timeout=5)

    assert worker["task"].done()


def test_worker_command_needs_an_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGRO_AWS_ENDPOINT_URL")
    get_settings.cache_clear()

    assert CliRunner().invoke(app, ["worker"]).exit_code == 1
