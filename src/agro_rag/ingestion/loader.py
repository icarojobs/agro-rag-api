import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceDocument:
    source: str
    title: str
    category: str
    body: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.title}\n{self.body}".encode()).hexdigest()


def parse_markdown(path: Path, root: Path) -> SourceDocument:
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        title, body = lines[0][2:].strip(), "\n".join(lines[1:]).strip()
    else:
        title, body = path.stem.replace("-", " ").capitalize(), text
    relative = path.relative_to(root)
    category = relative.parts[0] if len(relative.parts) > 1 else "geral"
    return SourceDocument(source=relative.as_posix(), title=title, category=category, body=body)


def load_corpus(root: Path) -> list[SourceDocument]:
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    return [parse_markdown(p, root) for p in sorted(root.rglob("*.md"))]
