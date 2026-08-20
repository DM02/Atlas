import pytest

from app.ai.pipeline.chunking import chunk_by_headings, chunk_pages
from app.ai.pipeline.extraction import ExtractedPage


def test_chunk_pages_splits_long_page_with_overlap() -> None:
    long_text = "word " * 1000  # far more than max_tokens
    pages = [ExtractedPage(page_number=1, text=long_text)]

    chunks = chunk_pages(pages, max_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
    assert all(c.page_number == 1 for c in chunks)
    # chunk_index is sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_pages_skips_blank_pages() -> None:
    pages = [
        ExtractedPage(page_number=1, text="   \n\n  "),
        ExtractedPage(page_number=2, text="content"),
    ]

    chunks = chunk_pages(pages, max_tokens=100, overlap_tokens=10)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_pages_rejects_overlap_not_smaller_than_max_tokens() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_pages([ExtractedPage(page_number=1, text="x")], max_tokens=50, overlap_tokens=50)


def test_chunk_pages_single_short_page_yields_one_chunk() -> None:
    pages = [ExtractedPage(page_number=1, text="short text")]

    chunks = chunk_pages(pages, max_tokens=100, overlap_tokens=10)

    assert len(chunks) == 1
    assert chunks[0].content == "short text"


def test_chunk_pages_never_sets_section_title() -> None:
    pages = [ExtractedPage(page_number=1, text="## Heading\n\nsome content")]

    chunks = chunk_pages(pages, max_tokens=100, overlap_tokens=10)

    assert all(c.section_title is None for c in chunks)


def test_chunk_by_headings_tags_each_chunk_with_its_section() -> None:
    text = (
        "## Vacation Policy\n\n"
        "Full-time employees accrue 15 days per year.\n\n"
        "## Remote Work Policy\n\n"
        "Employees may work remotely up to three days per week."
    )
    pages = [ExtractedPage(page_number=1, text=text)]

    chunks = chunk_by_headings(pages, max_tokens=100, overlap_tokens=10)

    assert [c.section_title for c in chunks] == ["Vacation Policy", "Remote Work Policy"]
    assert "15 days" in chunks[0].content
    assert "three days" in chunks[1].content


def test_chunk_by_headings_keeps_preamble_before_first_heading() -> None:
    text = "Intro paragraph with no heading.\n\n## Section One\n\nBody text."
    pages = [ExtractedPage(page_number=1, text=text)]

    chunks = chunk_by_headings(pages, max_tokens=100, overlap_tokens=10)

    assert chunks[0].section_title is None
    assert "Intro paragraph" in chunks[0].content
    assert chunks[1].section_title == "Section One"


def test_chunk_by_headings_falls_back_to_single_section_without_headings() -> None:
    pages = [ExtractedPage(page_number=1, text="Just plain text, no markdown headings.")]

    chunks = chunk_by_headings(pages, max_tokens=100, overlap_tokens=10)

    assert len(chunks) == 1
    assert chunks[0].section_title is None


def test_chunk_by_headings_splits_long_section_with_overlap() -> None:
    long_section = "## Big Section\n\n" + "word " * 1000
    pages = [ExtractedPage(page_number=1, text=long_section)]

    chunks = chunk_by_headings(pages, max_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    assert all(c.section_title == "Big Section" for c in chunks)
    assert all(c.token_count <= 100 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_by_headings_rejects_overlap_not_smaller_than_max_tokens() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_by_headings(
            [ExtractedPage(page_number=1, text="## H\n\nx")], max_tokens=50, overlap_tokens=50
        )
