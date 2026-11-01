"""parsers/common.py - helpers shared by all parsers (text cleanup, table-row formatting)."""
import re

Block = tuple[str, str, str]  # (kind: "para" | "table_row", text, style_name)


def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\d+)\.([^\s\d])", r"\1. \2", text)  # "13.Анонимизира" -> "13. Анонимизира"
    return text.strip()


def table_to_blocks(rows: list[list[str]], has_header: bool = True) -> list[Block]:
    """One block per body row: 'Header: value; Header: value' (rows[0] is the header).
    Without a header row, each row becomes 'cell | cell | cell'."""
    if not rows:
        return []
    if not has_header:
        return [("table_row", " | ".join(c for c in r if c), "") for r in rows if any(r)]
    header, body = rows[0], rows[1:]
    if not body:
        return [("table_row", " | ".join(c for c in header if c), "")]
    out = []
    for cells in body:
        if len(cells) == len(header):
            pairs = [f"{h}: {c}" if h else c for h, c in zip(header, cells) if c]
            text = "; ".join(pairs)
        else:
            text = " | ".join(c for c in cells if c)
        if text:
            out.append(("table_row", text, ""))
    return out