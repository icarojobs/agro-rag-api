"""Ingestion jobs: the message published to SNS and the DynamoDB record that tracks it."""

import hashlib
import time
import uuid
from typing import Any, Literal

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from agro_rag.aws import Infra, aws_client
from agro_rag.config import Settings

JobStatus = Literal["queued", "processing", "done", "failed"]

_serialize = TypeSerializer().serialize
_deserialize = TypeDeserializer().deserialize


class IngestionMessage(BaseModel):
    job_id: str
    collection: str
    source: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=50)
    content: str = Field(min_length=1, max_length=50_000)


class JobAlreadyDoneError(Exception):
    """The job finished before; a redelivered message must be acknowledged, not redone."""


def job_id_for(idempotency_key: str | None) -> str:
    """Same key, same job: a client retrying the POST does not enqueue the work twice."""
    if idempotency_key is None:
        return uuid.uuid4().hex
    return hashlib.sha256(idempotency_key.encode()).hexdigest()[:32]


def _item(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: _deserialize(v) for k, v in raw.items()}


def _is(exc: ClientError, code: str) -> bool:
    return bool(exc.response["Error"]["Code"] == code)


class JobStore:
    """Job status in DynamoDB. Conditional writes make every transition safe to repeat."""

    def __init__(self, settings: Settings, infra: Infra) -> None:
        self._settings = settings
        self._table = infra.table

    async def create(self, job_id: str, message: IngestionMessage) -> bool:
        """Returns False when a job with this id already exists."""
        item = {
            "job_id": job_id,
            "status": "queued",
            "attempts": 0,
            "source": message.source,
            "collection": message.collection,
            "created_at": int(time.time()),
        }
        async with aws_client(self._settings, "dynamodb") as ddb:
            try:
                await ddb.put_item(
                    TableName=self._table,
                    Item={k: _serialize(v) for k, v in item.items()},
                    ConditionExpression="attribute_not_exists(job_id)",
                )
            except ClientError as exc:
                if _is(exc, "ConditionalCheckFailedException"):
                    return False
                raise
        return True

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with aws_client(self._settings, "dynamodb") as ddb:
            response = await ddb.get_item(
                TableName=self._table, Key={"job_id": _serialize(job_id)}, ConsistentRead=True
            )
        return _item(response["Item"]) if "Item" in response else None

    async def delete(self, job_id: str) -> None:
        async with aws_client(self._settings, "dynamodb") as ddb:
            await ddb.delete_item(TableName=self._table, Key={"job_id": _serialize(job_id)})

    async def _update(
        self, job_id: str, expression: str, values: dict[str, Any], **kw: Any
    ) -> None:
        async with aws_client(self._settings, "dynamodb") as ddb:
            await ddb.update_item(
                TableName=self._table,
                Key={"job_id": _serialize(job_id)},
                UpdateExpression=expression,
                ExpressionAttributeValues={f":{k}": _serialize(v) for k, v in values.items()},
                **kw,
            )

    async def start(self, job_id: str) -> None:
        """queued/processing/failed -> processing, counting the attempt.

        Raises `JobAlreadyDoneError` if the job is already done.
        """
        try:
            await self._update(
                job_id,
                "SET #s = :processing, attempts = if_not_exists(attempts, :zero) + :one, "
                "updated_at = :now",
                {
                    "processing": "processing",
                    "zero": 0,
                    "one": 1,
                    "now": int(time.time()),
                    "done": "done",
                },
                ConditionExpression="attribute_not_exists(#s) OR #s <> :done",
                ExpressionAttributeNames={"#s": "status"},
            )
        except ClientError as exc:
            if _is(exc, "ConditionalCheckFailedException"):
                raise JobAlreadyDoneError(job_id) from exc
            raise

    async def complete(self, job_id: str, result: dict[str, int]) -> None:
        await self._update(
            job_id,
            "SET #s = :done, #r = :result, updated_at = :now REMOVE #e",
            {"done": "done", "result": result, "now": int(time.time())},
            ExpressionAttributeNames={"#s": "status", "#r": "result", "#e": "error"},
        )

    async def fail(self, job_id: str, error: str) -> None:
        await self._update(
            job_id,
            "SET #s = :failed, #e = :error, updated_at = :now",
            {"failed": "failed", "error": error[:500], "now": int(time.time())},
            ExpressionAttributeNames={"#s": "status", "#e": "error"},
        )
