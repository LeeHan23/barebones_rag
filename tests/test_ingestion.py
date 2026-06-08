"""Unit tests for text extraction and chunking (ingestion.py)."""
import pytest

from ingestion import SUPPORTED_EXTENSIONS, chunk_text, extract_text


def test_supported_extensions():
    assert SUPPORTED_EXTENSIONS == {".pdf", ".docx", ".txt", ".md"}


def test_extract_text_plain_decodes_utf8():
    assert extract_text(".txt", "hello world".encode("utf-8")) == "hello world"


def test_extract_text_plain_replaces_invalid_bytes():
    # 0xff is not valid UTF-8 — should be replaced, not raise
    text = extract_text(".txt", b"valid \xff bytes")
    assert "valid" in text
    assert "bytes" in text


def test_extract_text_md_uses_plain_decoder():
    assert extract_text(".md", b"# Heading\n\nbody") == "# Heading\n\nbody"


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_chunk_text_short_text_single_chunk():
    text = "one two three four five"
    chunks = chunk_text(text, size=800, overlap=100)
    assert chunks == [text]


def test_chunk_text_respects_size_and_overlap():
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)

    chunks = chunk_text(text, size=10, overlap=2)

    # First chunk is the first 10 words
    assert chunks[0] == " ".join(words[0:10])
    # Second chunk starts `size - overlap` words later, i.e. word index 8
    assert chunks[1] == " ".join(words[8:18])
    # Every word should be covered by at least one chunk
    covered = set(" ".join(chunks).split())
    assert covered == set(words)


def test_chunk_text_no_infinite_loop_when_overlap_equals_size():
    # start += size - overlap must stay positive or this would hang forever
    chunks = chunk_text("a b c d e f", size=2, overlap=2)
    assert chunks  # completes without hanging


@pytest.mark.parametrize("size,overlap", [(800, 100), (600, 80), (50, 0)])
def test_chunk_text_chunks_are_nonempty(size, overlap):
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=size, overlap=overlap)
    assert all(c.strip() for c in chunks)
