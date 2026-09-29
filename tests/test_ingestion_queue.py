import json
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from agro_rag.aws import Infra, aws_client
from agro_rag.config import Settings, get_settings
from agro_rag.ingestion.jobs import IngestionMessage, JobAlreadyDoneError, JobStore, job_id_for
from agro_rag.ingestion.queue import IngestionQueue, get_ingestion_queue
from agro_rag.main import create_app

needs_emulator = pytest.mark.skipif(
    not os.environ.get("AGRO_AWS_ENDPOINT_URL"), reason="needs an AWS emulator (floci)"
)

DOC = {
    "source": "solo/novo.md",
    "title": "Calagem em plantio direto",
    "category": "solo",
    "content": "A calagem em plantio direto é feita na superfície, sem incorporação.",
}


@pytest.fixture
async def api(isolated: Infra) -> AsyncIterator[AsyncClient]:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _messages(url: str, settings: Settings) -> list[dict[str, Any]]:
    async with aws_client(settings, "sqs") as sqs:
        response = await sqs.receive_message(
            QueueUrl=url, WaitTimeSeconds=3, MaxNumberOfMessages=10
        )
    return [json.loads(m["Body"]) for m in response.get("Messages", [])]


def test_job_id_is_stable_per_idempotency_key() -> None:
    assert job_id_for("abc") == job_id_for("abc")
    assert job_id_for("abc") != job_id_for("abd")
    assert job_id_for(None) != job_id_for(None)


@needs_emulator
async def test_post_enqueues_a_job_and_fans_out(api: AsyncClient, isolated: Infra) -> None:
    response = await api.post("/ingest/documents", json=DOC)

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    settings = get_settings()
    for url in (isolated.queue_url, isolated.audit_queue_url):
        (message,) = await _messages(url, settings)
        assert message["job_id"] == job_id
        assert message["source"] == "solo/novo.md"
        assert message["collection"] == "default"
    job = (await api.get(f"/ingest/jobs/{job_id}")).json()
    assert job["status"] == "queued"
    assert job["attempts"] == 0


@needs_emulator
async def test_repeated_idempotency_key_enqueues_once(api: AsyncClient, isolated: Infra) -> None:
    headers = {"Idempotency-Key": "upload-42"}

    first = await api.post("/ingest/documents", json=DOC, headers=headers)
    second = await api.post("/ingest/documents", json=DOC, headers=headers)

    assert first.status_code == second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    assert second.json()["status"] == "queued"
    assert len(await _messages(isolated.queue_url, get_settings())) == 1


@needs_emulator
async def test_unknown_job_is_404(api: AsyncClient) -> None:
    assert (await api.get("/ingest/jobs/nope")).status_code == 404


@needs_emulator
async def test_invalid_payload_is_422(api: AsyncClient) -> None:
    assert (await api.post("/ingest/documents", json={**DOC, "content": ""})).status_code == 422


async def test_endpoint_is_503_when_aws_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGRO_AWS_ENDPOINT_URL", raising=False)
    get_settings.cache_clear()
    get_ingestion_queue.cache_clear()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        post = await c.post("/ingest/documents", json=DOC)
        get = await c.get("/ingest/jobs/x")

    assert post.status_code == get.status_code == 503


async def test_endpoint_is_503_when_the_emulator_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGRO_AWS_ENDPOINT_URL", "http://127.0.0.1:1")
    get_settings.cache_clear()
    get_ingestion_queue.cache_clear()
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/ingest/documents", json=DOC)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "10"


@needs_emulator
async def test_failed_publish_removes_the_job(
    isolated: Infra, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    queue = IngestionQueue(settings)
    message = IngestionMessage(job_id="", collection="default", **DOC)

    class Boom(Exception): ...

    def failing(settings: Settings, service: str) -> Any:
        if service == "sns":
            raise Boom
        return aws_client(settings, service)

    monkeypatch.setattr("agro_rag.ingestion.queue.aws_client", failing)

    with pytest.raises(Boom):
        await queue.enqueue(message, idempotency_key="k1")

    assert await JobStore(settings, isolated).get(job_id_for("k1")) is None


@needs_emulator
async def test_job_store_transitions(isolated: Infra) -> None:
    store = JobStore(get_settings(), isolated)
    message = IngestionMessage(job_id="j1", collection="default", **DOC)

    assert await store.create("j1", message) is True
    assert await store.create("j1", message) is False
    await store.start("j1")
    await store.fail("j1", "boom")
    await store.start("j1")
    failed_then_retried = await store.get("j1")
    await store.complete("j1", {"created": 1})
    done = await store.get("j1")

    assert failed_then_retried is not None
    assert failed_then_retried["attempts"] == 2
    assert failed_then_retried["error"] == "boom"
    assert done is not None
    assert done["status"] == "done"
    assert done["result"] == {"created": 1}
    assert "error" not in done
    with pytest.raises(JobAlreadyDoneError):
        await store.start("j1")
    await store.delete("j1")
    assert await store.get("j1") is None
