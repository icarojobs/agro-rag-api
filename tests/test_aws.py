import json
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from typer.testing import CliRunner

from agro_rag.aws import Infra, aws_client, aws_enabled, ensure_infrastructure
from agro_rag.cli import app
from agro_rag.config import Settings, get_settings

needs_emulator = pytest.mark.skipif(
    not os.environ.get("AGRO_AWS_ENDPOINT_URL"), reason="needs an AWS emulator (floci)"
)


@pytest.fixture
def settings() -> Settings:
    # Unique names per test session keep runs isolated from a shared emulator.
    return Settings(
        ingestion_topic="t-infra",
        ingestion_queue="t-infra-jobs",
        ingestion_dlq="t-infra-dlq",
        ingestion_audit_queue="t-infra-audit",
        ingestion_table="t-infra-table",
        ingestion_max_receive_count=2,
    )


@pytest.fixture
async def infra(settings: Settings) -> Infra:
    return await ensure_infrastructure(settings)


@pytest.fixture
async def sqs(settings: Settings) -> AsyncIterator[Any]:
    async with aws_client(settings, "sqs") as client:
        yield client


@needs_emulator
async def test_ensure_infrastructure_is_idempotent(settings: Settings, infra: Infra) -> None:
    again = await ensure_infrastructure(settings)

    assert again == infra
    assert infra.table == "t-infra-table"


@needs_emulator
async def test_queue_has_redrive_policy_and_visibility_timeout(
    sqs: Any, infra: Infra, settings: Settings
) -> None:
    attrs = (await sqs.get_queue_attributes(QueueUrl=infra.queue_url, AttributeNames=["All"]))[
        "Attributes"
    ]
    dlq_arn = (await sqs.get_queue_attributes(QueueUrl=infra.dlq_url, AttributeNames=["QueueArn"]))[
        "Attributes"
    ]["QueueArn"]

    redrive = json.loads(attrs["RedrivePolicy"])
    assert redrive["deadLetterTargetArn"] == dlq_arn
    assert int(redrive["maxReceiveCount"]) == 2
    assert attrs["VisibilityTimeout"] == str(settings.ingestion_visibility_timeout_seconds)


@needs_emulator
async def test_sns_fans_out_to_both_queues(sqs: Any, infra: Infra, settings: Settings) -> None:
    async with aws_client(settings, "sns") as sns:
        await sns.publish(TopicArn=infra.topic_arn, Message=json.dumps({"hello": "fanout"}))

    for url in (infra.queue_url, infra.audit_queue_url):
        received = await sqs.receive_message(QueueUrl=url, WaitTimeSeconds=3)
        assert json.loads(received["Messages"][0]["Body"]) == {"hello": "fanout"}
        await sqs.delete_message(
            QueueUrl=url, ReceiptHandle=received["Messages"][0]["ReceiptHandle"]
        )


@needs_emulator
async def test_table_exists_with_job_id_key(settings: Settings, infra: Infra) -> None:
    async with aws_client(settings, "dynamodb") as ddb:
        table = (await ddb.describe_table(TableName=infra.table))["Table"]

    assert table["KeySchema"] == [{"AttributeName": "job_id", "KeyType": "HASH"}]


def test_aws_enabled_follows_the_endpoint() -> None:
    assert aws_enabled(Settings(aws_endpoint_url="http://x:4566"))
    assert not aws_enabled(Settings(aws_endpoint_url=None))


@needs_emulator
def test_aws_init_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGRO_INGESTION_TOPIC", "t-cli")
    monkeypatch.setenv("AGRO_INGESTION_QUEUE", "t-cli-jobs")
    monkeypatch.setenv("AGRO_INGESTION_DLQ", "t-cli-dlq")
    monkeypatch.setenv("AGRO_INGESTION_AUDIT_QUEUE", "t-cli-audit")
    monkeypatch.setenv("AGRO_INGESTION_TABLE", "t-cli-table")
    get_settings.cache_clear()

    result = CliRunner().invoke(app, ["aws-init"])

    assert result.exit_code == 0, result.output
    assert "table=t-cli-table" in result.output


def test_aws_init_fails_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGRO_AWS_ENDPOINT_URL", raising=False)
    get_settings.cache_clear()

    result = CliRunner().invoke(app, ["aws-init"])

    assert result.exit_code == 1
