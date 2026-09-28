import json
from pathlib import Path

import mlflow
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from agro_rag.cli import app
from agro_rag.embeddings import HashingEmbeddings
from agro_rag.evaluation.runner import EvalConfig, evaluate_config, load_questions, log_to_mlflow


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    path = tmp_path / "questions.jsonl"
    rows = [
        {
            "id": "q1",
            "question": "como o calcário corrige a acidez?",
            "relevant": ["solo/calagem.md"],
        },
        {
            "id": "q2",
            "question": "percevejo nos grãos de soja",
            "relevant": ["pragas/percevejo.md"],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    return path


def test_collection_name_is_derived_from_config() -> None:
    config = EvalConfig(chunk_size=300, chunk_overlap=50, embedding_model="org/My_Model-v2")

    assert config.collection == "eval-my-model-v2-300-50"


async def test_evaluate_config_and_log_to_mlflow(
    session: AsyncSession, corpus_dir: Path, questions_file: Path, tmp_path: Path
) -> None:
    questions = load_questions(questions_file)
    config = EvalConfig(chunk_size=200, chunk_overlap=20, ks=(1, 2))

    result = await evaluate_config(session, HashingEmbeddings(), corpus_dir, questions, config)
    again = await evaluate_config(session, HashingEmbeddings(), corpus_dir, questions, config)

    assert result.summary["recall@1"] == 1.0
    assert result.summary["mrr"] == 1.0
    assert result.params["n_chunks"] == again.params["n_chunks"] > 0
    assert list(result.per_question["id"]) == ["q1", "q2"]

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = log_to_mlflow(result, uri, "test-exp")
    run = mlflow.get_run(run_id)
    assert run.data.metrics["recall_at_1"] == 1.0
    assert run.data.params["chunk_size"] == "200"


def test_cli_evaluate(corpus_dir: Path, questions_file: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--questions",
            str(questions_file),
            "--corpus",
            str(corpus_dir),
            "--chunk-sizes",
            "150,300",
            "--k",
            "1,2",
            "--mlflow-uri",
            "",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "recall@1" in result.output
    assert "mrr" in result.output
