"""AWS clients and the messaging infrastructure used by asynchronous ingestion.

POST /ingest/documents -> SNS topic -+-> SQS queue (worker) --(after N failed receives)--> DLQ
                                     +-> SQS audit queue (second subscriber)
DynamoDB table: one item per job (status, attempts, error)
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aioboto3
import structlog
from botocore.config import Config
from botocore.exceptions import ClientError

from agro_rag.config import Settings

logger = structlog.get_logger(__name__)

# The worker long-polls for up to 20 s, so the read timeout has to be longer than that.
_BOTO_CONFIG = Config(
    connect_timeout=2, read_timeout=30, retries={"max_attempts": 3, "mode": "standard"}
)


@dataclass(frozen=True, slots=True)
class Infra:
    topic_arn: str
    queue_url: str
    dlq_url: str
    audit_queue_url: str
    table: str


def aws_enabled(settings: Settings) -> bool:
    return bool(settings.aws_endpoint_url)


def make_session(settings: Settings) -> aioboto3.Session:
    return aioboto3.Session(
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )


@asynccontextmanager
async def aws_client(settings: Settings, service: str) -> AsyncIterator[Any]:
    async with make_session(settings).client(
        service,
        endpoint_url=settings.aws_endpoint_url,
        config=_BOTO_CONFIG,
    ) as client:
        yield client


async def _queue_arn(sqs: Any, url: str) -> str:
    attrs = await sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["QueueArn"])
    return str(attrs["Attributes"]["QueueArn"])


def _allow_topic_policy(queue_arn: str, topic_arn: str) -> str:
    """Queue policy that lets the SNS topic deliver messages (required on real AWS)."""
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sns.amazonaws.com"},
                    "Action": "sqs:SendMessage",
                    "Resource": queue_arn,
                    "Condition": {"ArnEquals": {"aws:SourceArn": topic_arn}},
                }
            ],
        }
    )


async def ensure_infrastructure(settings: Settings) -> Infra:
    """Create the topic, queues, subscriptions and table. Safe to run repeatedly."""
    async with (
        aws_client(settings, "sqs") as sqs,
        aws_client(settings, "sns") as sns,
        aws_client(settings, "dynamodb") as dynamodb,
    ):
        dlq_url = (await sqs.create_queue(QueueName=settings.ingestion_dlq))["QueueUrl"]
        dlq_arn = await _queue_arn(sqs, dlq_url)
        topic_arn = (await sns.create_topic(Name=settings.ingestion_topic))["TopicArn"]

        async def subscribed_queue(name: str, attributes: dict[str, str]) -> str:
            url = (await sqs.create_queue(QueueName=name, Attributes=attributes))["QueueUrl"]
            arn = await _queue_arn(sqs, url)
            await sqs.set_queue_attributes(
                QueueUrl=url, Attributes={"Policy": _allow_topic_policy(arn, topic_arn)}
            )
            await sns.subscribe(
                TopicArn=topic_arn,
                Protocol="sqs",
                Endpoint=arn,
                Attributes={"RawMessageDelivery": "true"},
            )
            return str(url)

        queue_url = await subscribed_queue(
            settings.ingestion_queue,
            {
                "VisibilityTimeout": str(settings.ingestion_visibility_timeout_seconds),
                "RedrivePolicy": json.dumps(
                    {
                        "deadLetterTargetArn": dlq_arn,
                        "maxReceiveCount": str(settings.ingestion_max_receive_count),
                    }
                ),
            },
        )
        audit_url = await subscribed_queue(settings.ingestion_audit_queue, {})

        try:
            await dynamodb.create_table(
                TableName=settings.ingestion_table,
                KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceInUseException":
                raise

    logger.info("aws_infrastructure_ready", queue=settings.ingestion_queue)
    return Infra(
        topic_arn=topic_arn,
        queue_url=queue_url,
        dlq_url=str(dlq_url),
        audit_queue_url=audit_url,
        table=settings.ingestion_table,
    )
