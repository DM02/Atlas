from app.ai.pipeline.cleaning import clean_text


def test_clean_text_collapses_whitespace_and_blank_lines() -> None:
    raw = "Title\r\n\r\n\r\n\r\nBody   text\twith\ttabs\r\n\r\nmore"

    result = clean_text(raw)

    assert result == "Title\n\nBody text with tabs\n\nmore"


def test_clean_text_strips_leading_trailing_whitespace() -> None:
    assert clean_text("   padded text   \n") == "padded text"
