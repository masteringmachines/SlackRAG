"""Vector store + ingestion. Knowledge bases are scoped per Slack channel (or DM)."""
import hashlib
import re
from pathlib import Path

import chromadb

import config


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    """Paragraph-aware chunking with character overlap; oversized paragraphs are hard-split."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    pieces = []
    for p in (p.strip() for p in text.split("\n\n")):
        while len(p) > size:
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            pieces.append(p[:cut].strip())
            p = p[max(cut - overlap, 1):].strip()
        if p:
            pieces.append(p)
    chunks, buf = [], ""
    for p in pieces:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) > 20]


class KnowledgeBase:
    def __init__(self, path: str = config.CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=path)
        # Default embedding function: local MiniLM (ONNX). No extra API key required.
        self.col = self.client.get_or_create_collection("kb", metadata={"hnsw:space": "cosine"})

    def add_document(self, scope: str, source: str, text: str) -> int:
        chunks = chunk_text(text)
        if not chunks:
            return 0
        self.delete_source(scope, source)  # re-upload replaces the old version
        prefix = hashlib.sha1(f"{scope}:{source}".encode()).hexdigest()[:12]
        self.col.add(
            ids=[f"{prefix}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"scope": scope, "source": source, "chunk": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def search(self, scope: str, query: str, k: int = config.TOP_K):
        res = self.col.query(query_texts=[query], n_results=k, where={"scope": scope})
        return [
            {"source": m["source"], "chunk": m["chunk"], "score": round(1 - d, 3), "text": doc}
            for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def list_sources(self, scope: str) -> dict[str, int]:
        got = self.col.get(where={"scope": scope}, include=["metadatas"])
        counts: dict[str, int] = {}
        for m in got["metadatas"]:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def delete_source(self, scope: str, source: str) -> None:
        self.col.delete(where={"$and": [{"scope": scope}, {"source": source}]})
