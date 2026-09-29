import asyncio
from functools import lru_cache

from agro_rag.aws import Infra, aws_client, aws_enabled, ensure_infrastructure
from agro_rag.config import Settings, get_settings
from agro_rag.ingestion.jobs import IngestionMessage, JobStore, job_id_for


class IngestionQueue:
    """Publishes ingestion jobs to the SNS topic and reads their status."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._infra: Infra | None = None
        self._lock = asyncio.Lock()

    async def _resolve(self) -> Infra:
        # Resolved on first use (not at startup) so the API boots without the queue.
        async with self._lock:
            if self._infra is None:
                self._infra = await ensure_infrastructure(self._settings)
            return self._infra

    async def enqueue(
        self, message: IngestionMessage, *, idempotency_key: str | None = None
    ) -> tuple[str, bool]:
        """Returns `(job_id, created)`; `created` is False for a repeated idempotency key."""
        infra = await self._resolve()
        job_id = job_id_for(idempotency_key)
        message = message.model_copy(update={"job_id": job_id})
        store = JobStore(self._settings, infra)
        if not await store.create(job_id, message):
            return job_id, False
        try:
            async with aws_client(self._settings, "sns") as sns:
                await sns.publish(TopicArn=infra.topic_arn, Message=message.model_dump_json())
        except Exception:
            await store.delete(job_id)
            raise
        return job_id, True

    async def job(self, job_id: str) -> dict[str, object] | None:
        return await JobStore(self._settings, await self._resolve()).get(job_id)


@lru_cache
def get_ingestion_queue() -> IngestionQueue | None:
    settings = get_settings()
    return IngestionQueue(settings) if aws_enabled(settings) else None
