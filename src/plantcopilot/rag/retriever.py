"""Document retrieval over manuals and maintenance reports.

Documents are split at markdown headings, so every chunk is one section with a
stable id (file#section). Scoring is TF-IDF with word bigrams. The interface
(`search(query, k) -> list[Chunk]`) is what the agent sees, so this can be
replaced by a vector index such as Azure AI Search without touching the agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    score: float = 0.0


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_markdown(folder: str | Path) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        doc_title, section, buf = path.stem, "intro", []

        def flush():
            body = " ".join(buf).strip()
            if body:
                chunks.append(Chunk(f"{path.stem}#{_slug(section)}", f"{doc_title}: {section}", body))

        for line in path.read_text().splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                section, buf = line[3:].strip(), []
            elif line.strip():
                buf.append(line.strip("- ").strip())
        flush()
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.X = self.vec.fit_transform([f"{c.title} {c.text}" for c in chunks])

    @classmethod
    def from_folder(cls, folder: str | Path) -> "Retriever":
        return cls(load_markdown(folder))

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        q = self.vec.transform([query])
        scores = (self.X @ q.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [Chunk(self.chunks[i].id, self.chunks[i].title, self.chunks[i].text, round(float(scores[i]), 3))
                for i in top if scores[i] > 0]
