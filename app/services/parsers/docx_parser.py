"""parsers/docx_parser.py - .docx -> ordered blocks (paragraphs + tables). No chunking, no embeddings, no DB."""
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .common import Block, clean, table_to_blocks


def _table_blocks(table: Table) -> list[Block]:
    rows = []
    for row in table.rows:
        cells, prev = [], None
        for cell in row.cells:  # merged cells repeat -> drop adjacent duplicates
            t = clean(cell.text)
            if t and t != prev:
                cells.append(t)
            prev = t
        if cells:
            rows.append(cells)
    return table_to_blocks(rows)


def read_blocks(path: str) -> list[Block]:
    """Paragraphs AND tables, in the order they appear (doc.paragraphs skips tables)."""
    doc = Document(path)
    blocks: list[Block] = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, doc)
            text = clean(p.text)
            if text:
                try:
                    style = p.style.name or ""
                except Exception:
                    style = ""
                blocks.append(("para", text, style))
        elif child.tag.endswith("}tbl"):
            blocks.extend(_table_blocks(Table(child, doc)))
    return blocks