"""parsers - one module per file format, all returning the same list of blocks."""
import os

from .common import Block
from .docx_parser import read_blocks as _read_docx
from .pdf_parser import read_blocks as _read_pdf

PARSERS = {".docx": _read_docx, ".pdf": _read_pdf}
SUPPORTED = tuple(PARSERS)


def read_blocks(path: str) -> list[Block]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext} (supported: {', '.join(SUPPORTED)})")
    return PARSERS[ext](path)