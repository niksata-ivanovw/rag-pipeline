"""
parsers/pdf_parser.py - .pdf -> ordered blocks, using pdfplumber. No chunking, embeddings or DB.

What it does that plain extract_text() does not:
  * tables are found first and CROPPED OUT of the page text, so nothing appears twice;
  * hard-wrapped lines are rejoined into paragraphs (by line spacing / short last line / bullets);
  * headings are detected from font size and bold, and emitted with style "Heading"
    (chunking.heading_level() already treats that style as a section boundary);
  * repeated headers/footers and bare page numbers are dropped;
  * dotted form leaders ("Име……………") are collapsed.

Limits: scanned PDFs have no text layer (needs OCR); multi-row table headers and
multi-column page layouts are not understood.
"""
import re
import statistics as st

import pdfplumber

from .common import Block, clean, table_to_blocks

TERMINAL = tuple(".!?:;…»“”\"")
BULLET = re.compile(r"^([-–—•▪*=]|\d{1,2}[.)]|[а-яА-Яa-zA-Z][.)/])\s")
LEADER = re.compile(r"(?:[.…]\s*){3,}")


def _is_inside(c: dict, boxes: list[tuple]) -> bool:
    return any(c["x0"] >= x0 - 1 and c["x1"] <= x1 + 1 and c["top"] >= t - 1 and c["bottom"] <= b + 1
               for x0, t, x1, b in boxes)


def _line_info(line: dict) -> dict:
    chars = [c for c in line["chars"] if c["text"].strip()]
    sizes = [c["size"] for c in chars] or [0]
    bold = sum("bold" in c["fontname"].lower() for c in chars) / max(len(chars), 1)
    return {"text": line["text"].strip(), "x0": line["x0"], "x1": line["x1"], "top": line["top"],
            "bottom": line["bottom"], "size": st.median(sizes), "bold": bold >= 0.6, "n": len(chars)}


def _norm_rows(table) -> list[list[str]]:
    rows, last_first = [], ""
    for raw in table.extract():
        cells = [clean(c or "") for c in raw]
        if cells and cells[0]:
            last_first = cells[0]
        elif cells and any(cells[1:]):
            cells[0] = last_first  # merged first-column cell -> repeat the label
        if any(cells[1:]) or (cells and cells[0] and not cells[0].isdigit()):
            rows.append(cells)  # drops blank form rows like ["1", "", "", ...]
        elif len(rows) == 0:
            rows.append(cells)
    return rows


def _bold_ratio(chars: list[dict]) -> float:
    cs = [c for c in chars if c["text"].strip()]
    return sum("bold" in c["fontname"].lower() for c in cs) / len(cs) if cs else 0.0


def _has_header(page, table) -> bool:
    """A real header row is (almost) all bold. If the table has no bold text at all,
    assume the first row is the header; if only some cells are bold (row headers), it is data."""
    try:
        first = _bold_ratio(page.crop(table.rows[0].bbox).chars)
        whole = _bold_ratio(page.crop(table.bbox).chars)
    except Exception:
        return True
    return first >= 0.9 or whole == 0


def _page_items(page, boxes):
    """Lines of text (outside tables) and the tables themselves, as (top, kind, payload)."""
    text_page = page.filter(lambda o: o.get("object_type") != "char" or not _is_inside(o, boxes))
    items = [(l["top"], "line", _line_info(l)) for l in text_page.extract_text_lines(return_chars=True)]
    return items


def read_blocks(path: str) -> list[Block]:
    with pdfplumber.open(path) as pdf:
        per_page = []
        for pno, page in enumerate(pdf.pages):
            tables = page.find_tables()
            boxes = [t.bbox for t in tables]
            items = _page_items(page, boxes)
            items += [(t.bbox[1], "table", table_to_blocks(_norm_rows(t), _has_header(page, t))) for t in tables]
            items.sort(key=lambda x: x[0])
            per_page.append((pno, page.height, page.width, items))

    lines = [p for _, _, _, items in per_page for _, k, p in items if k == "line"]
    if not lines:
        return []  # no text layer -> scanned PDF
    body = st.median([l["size"] for l in lines for _ in range(l["n"])])  # weighted by characters
    line_h = st.median([l["bottom"] - l["top"] for l in lines])

    # repeated header/footer text (digits normalised), only meaningful for >= 3 pages
    seen: dict[str, int] = {}
    for _, h, _, items in per_page:
        for top, k, p in items:
            if k == "line" and (top < h * 0.07 or p["bottom"] > h * 0.93):
                key = re.sub(r"\d+", "#", p["text"])
                seen[key] = seen.get(key, 0) + 1
    repeated = {k for k, v in seen.items() if len(per_page) >= 3 and v >= len(per_page) / 2}

    blocks: list[Block] = []
    buf: list[str] = []
    prev = None  # previous kept line (same paragraph candidate)

    def flush():
        nonlocal buf, prev
        if buf:
            text = clean(LEADER.sub(" … ", " ".join(buf)))
            if re.search(r"\w", text):
                blocks.append(("para", text, ""))
        buf, prev = [], None

    for pno, h, w, items in per_page:
        right = sorted(p["x1"] for _, k, p in items if k == "line")
        margin = right[int(len(right) * 0.9)] if right else w
        for top, kind, payload in items:
            if kind == "table":
                flush()
                blocks.extend(payload)
                continue
            line = payload
            edge = top < h * 0.07 or line["bottom"] > h * 0.93
            if edge and (re.sub(r"\d+", "#", line["text"]) in repeated or re.fullmatch(r"\W*\d+\W*", line["text"])):
                continue
            is_heading = (line["size"] >= body * 1.15 or (line["bold"] and len(line["text"]) <= 120
                          and not line["text"].endswith(TERMINAL[:3]))) and len(line["text"]) <= 150
            if is_heading:
                flush()
                if blocks and blocks[-1][2] == "Heading":  # multi-line heading -> one block
                    blocks[-1] = ("para", clean(blocks[-1][1] + " " + line["text"]), "Heading")
                else:
                    blocks.append(("para", clean(line["text"]), "Heading"))
                continue
            if prev is not None:
                same_page = prev["_p"] == pno
                gap_break = same_page and (line["top"] - prev["bottom"]) > 0.5 * line_h
                short_end = prev["text"].endswith(TERMINAL) and prev["x1"] < margin - 0.12 * w
                if gap_break or short_end or BULLET.match(line["text"]):
                    flush()
            text = line["text"]
            if buf and buf[-1].endswith("-") and text[:1].islower() and len(buf[-1]) > 1 and buf[-1][-2].isalpha():
                buf[-1] = buf[-1][:-1] + text  # re-join a hyphenated word split across lines
            else:
                buf.append(text)
            prev = {**line, "_p": pno}
    flush()
    return blocks