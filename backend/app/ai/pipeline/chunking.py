import re
from dataclasses import dataclass

import tiktoken

from app.ai.pipeline.cleaning import clean_text
from app.ai.pipeline.extraction import ExtractedPage

_ENCODING = tiktoken.get_encoding("cl100k_base")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    content: str
    page_number: int | None
    chunk_index: int
    token_count: int
    section_title: str | None = None


def chunk_pages(pages: list[ExtractedPage], max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Baseline chunking strategy: fixed-size token windows with overlap, per page.

    Chunk boundaries ignore sentence/paragraph structure — this is deliberately the
    naive baseline that Phase 5's chunking experiment compares against a
    structure-aware strategy (see chunk_by_headings() below, docs/ARCHITECTURE.md
    §7, §10).
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue

        tokens = _ENCODING.encode(cleaned)
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(
                Chunk(
                    content=_ENCODING.decode(chunk_tokens),
                    page_number=page.page_number,
                    chunk_index=index,
                    token_count=len(chunk_tokens),
                )
            )
            index += 1
            if end == len(tokens):
                break
            start = end - overlap_tokens

    return chunks


def chunk_by_headings(
    pages: list[ExtractedPage], max_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    """Structure-aware chunking strategy: splits on markdown-style '## Heading'
    section boundaries first, then applies the same token-window logic within
    each section so no chunk exceeds max_tokens. Every resulting chunk is
    tagged with the section_title of the heading it fell under (chunk_pages()
    never sets this — it has no notion of sections).

    This is Phase 5's second chunking strategy, required for the chunking
    comparison experiment (docs/ARCHITECTURE.md §10, docs/EVALUATION.md).
    Deliberately eval-only for now: the production ingestion pipeline keeps
    using chunk_pages() until the evaluation results say otherwise, matching
    this repo's "new capability, config-gated, not silently swapped in"
    stance already used for hybrid search and reranking.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue

        for section_title, section_text in _split_sections(cleaned):
            tokens = _ENCODING.encode(section_text)
            if not tokens:
                continue

            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunks.append(
                    Chunk(
                        content=_ENCODING.decode(chunk_tokens),
                        page_number=page.page_number,
                        chunk_index=index,
                        token_count=len(chunk_tokens),
                        section_title=section_title,
                    )
                )
                index += 1
                if end == len(tokens):
                    break
                start = end - overlap_tokens

    return chunks


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split text on '## Heading' markers into (section_title, body) pairs.

    Text preceding the first heading (if any) is kept under section_title=None
    rather than dropped. Text with no headings at all becomes a single
    (None, text) section, so this strategy degrades gracefully to
    "whole-page chunking" on unstructured documents.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((title, body))

    return sections
