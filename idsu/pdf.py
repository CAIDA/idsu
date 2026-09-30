"""Paper text extraction."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def pdf_text(path: str | Path) -> str:
    """The text the extraction prompt sees for one paper.

    load_and_split() chunks each page with LangChain's default splitter, whose
    chunks overlap, so some text appears twice. That is kept deliberately: it is
    the text the reported runs were given.
    """
    try:
        pages = PyPDFLoader(str(path)).load_and_split()
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF '{path}': {e}") from e
    return "".join(page.page_content + "\n\n" for page in pages)
