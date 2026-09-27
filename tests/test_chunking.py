from pathlib import Path

import pytest

from agro_rag.ingestion.chunking import build_splitter, chunk_document
from agro_rag.ingestion.loader import SourceDocument, load_corpus, parse_markdown


def test_load_corpus_parses_title_category_and_source(corpus_dir: Path) -> None:
    docs = load_corpus(corpus_dir)

    assert [d.source for d in docs] == ["pragas/percevejo.md", "solo/calagem.md"]
    calagem = docs[1]
    assert calagem.title == "Calagem"
    assert calagem.category == "solo"
    assert calagem.body.startswith("O calcário")


def test_parse_markdown_without_heading_uses_file_name(tmp_path: Path) -> None:
    path = tmp_path / "rotacao-de-culturas.md"
    path.write_text("Texto sem título.", encoding="utf-8")

    doc = parse_markdown(path, tmp_path)

    assert doc.title == "Rotacao de culturas"
    assert doc.category == "geral"


def test_load_corpus_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")


def test_content_hash_changes_with_body() -> None:
    a = SourceDocument("a.md", "T", "c", "um")
    b = SourceDocument("a.md", "T", "c", "dois")
    assert a.content_hash != b.content_hash


def test_chunk_document_respects_size_and_prefixes_title() -> None:
    body = " ".join(f"Frase número {i} sobre fertilidade do solo." for i in range(40))
    doc = SourceDocument("solo/x.md", "Fertilidade", "solo", body)

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 5
    assert all(c.startswith("Fertilidade\n") for c in chunks)
    assert all(len(c) <= 200 + len("Fertilidade\n") for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        build_splitter(100, 100)
