from langchain_text_splitters import RecursiveCharacterTextSplitter

from agro_rag.ingestion.loader import SourceDocument


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator="end",
    )


def chunk_document(doc: SourceDocument, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split the body and prefix every chunk with the document title for extra context."""
    splitter = build_splitter(chunk_size, chunk_overlap)
    return [f"{doc.title}\n{piece.strip()}" for piece in splitter.split_text(doc.body)]
