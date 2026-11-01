"""
chunking.py - structure-first, semantic-second chunking. Input = blocks; it never opens files.

Why not SemanticSplitterNodeParser alone?
  * its breakpoint is a *percentile*, so ~10 % of the gaps are ALWAYS cut (at 90),
    whether or not the topic really changes, and there is no min/max chunk size;
  * it joins sentences with "" (no space) -> "гр.Момчилград" in text and embeddings;
  * it only sees sentences, so headings, list items and tables are invisible to it.

Pipeline
  blocks (from a parser: paragraphs + tables, in order)
       -> sections (split on headings = hard boundaries)
       -> units (sentences, list items, table rows)
       -> windowed embeddings -> semantic breakpoints (only inside long sections)
       -> merge tiny / split huge chunks -> Chunk(content, embed_text, metadata)

Usage (with your existing Ollama model):
    from llama_index.embeddings.ollama import OllamaEmbedding
    em = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434")
    blocks = read_blocks("documents/obqva.docx")          # parsers/docx_parser.py
    chunks = chunk_blocks(blocks, em.get_text_embedding_batch, file_name="obqva.docx")
    # store c.content in `content`, embed c.embed_text, keep c.metadata as JSON.

Command line (compare settings quickly, no database needed):
    python chunking.py ../documents            # uses Ollama bge-m3
    python chunking.py ../documents --stub     # offline smoke test, NOT real quality
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

EmbedFn = Callable[[list[str]], list[list[float]]]
Block = tuple[str, str, str]  # (kind: "para" | "table_row", text, style_name) - produced by a parser


# --------------------------------------------------------------------------- #
# 2. Headings and sections                                                     #
# --------------------------------------------------------------------------- #
def heading_level(text: str, style: str) -> str | None:
    """'major', 'minor' or None. Heuristic - extend it for the patterns in YOUR documents."""
    if style.lower().startswith(("heading", "title", "заглавие")):
        return "major"
    if len(text) > 120:
        return None
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        return "major"  # "1. ИСТОРИЧЕСКИ КОНТЕКСТ", "О Б Я В А", "...„КОЛКО СИ ХУБАВА!“"
    if re.search(r'[.!?]["»“”]?$', text):
        return None  # looks like a normal sentence
    if len(text) <= 60 and re.match(r"^\d{1,2}\s*[.)]\s*\S", text):
        return "major"  # "6. Тема и лирически мотиви(:)"
    if text.endswith(":"):
        return "minor"
    return None


# "2. История на творбата – написана е през 1913 г. ..." -> section "2. История на творбата"
_LEAD = re.compile(r"^(\d{1,2})\s*\.\s*([^–—\-]{3,60}?)\s*[–—-]\s")


def _label(text: str) -> str:
    return re.sub(r"^[=\-–•\s]+", "", text).rstrip(": ")


# --------------------------------------------------------------------------- #
# 3. Sentence / list splitting                                                 #
# --------------------------------------------------------------------------- #
# A period after one of these is NOT a sentence end. "г." and "в." are included on
# purpose: "През 1913 г. Дебелянов..." continues the sentence.
ABBR = {
    "гр", "чл", "ал", "т", "ул", "бр", "ст", "стр", "напр", "вж", "сп", "др", "проф", "доц",
    "св", "пр", "тел", "ет", "ап", "кв", "вх", "обл", "общ", "мл", "ген", "д-р", "г-н", "г-жа",
    "т.н", "т.е", "т.нар", "т.г", "г", "гг", "в", "вв", "ч", "лв", "млн", "млрд", "и.д", "зам",
}
_BOUND = re.compile(r"([.!?…]+[»“”\"')\]]*)(\s+)")
_MARK = re.compile(r"(?:(?<=\s)|^)(\d{1,2})\s*\.\s*(?=[^\s\d])")


def explode_numbered(text: str) -> list[str]:
    """'... 1. a; 2. b; 3. c' inside ONE paragraph -> separate items (only if 1,2,3.. in order)."""
    chain, want = [], 1
    for m in _MARK.finditer(text):
        if int(m.group(1)) == want:
            chain.append(m.start())
            want += 1
    if len(chain) < 3:
        return [text]
    cuts = [0] + chain + [len(text)]
    return [p for p in (text[a:b].strip() for a, b in zip(cuts, cuts[1:])) if p]


def split_sentences(text: str) -> list[str]:
    out, start = [], 0
    for m in _BOUND.finditer(text):
        nxt = text[m.end(): m.end() + 1]
        if not nxt or nxt.islower():  # next word is lowercase -> same sentence
            continue
        if m.group(1).startswith("."):
            tok = re.search(r"[\w\-.]+$", text[start:m.start(1)])
            word = tok.group(0).strip(".").lower() if tok else ""
            if word in ABBR or (word.isdigit() and len(word) <= 2):  # abbreviation / list number
                continue
            if len(word) == 1 and word.isalpha():  # initials: "М. К.", "Д. Дебелянов"
                continue
        out.append(text[start:m.end(1)].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if re.search(r"\w", s)]  # drops "!!!" style fragments


# --------------------------------------------------------------------------- #
# 4. Chunking                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    text: str
    block: int  # index of the source block, to keep line breaks between paragraphs


@dataclass
class Chunk:
    content: str  # clean text: store it, show it to the LLM
    embed_text: str  # "title > section\ncontent": embed THIS
    section: str
    index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class _Section:
    label: str
    units: list[Unit] = field(default_factory=list)


def build_sections(blocks: list[Block]) -> list[_Section]:
    sections, major, minor = [_Section("")], "", ""

    def open_section(label: str) -> None:
        if sections[-1].units:
            sections.append(_Section(label))
        else:  # consecutive headings -> same section, just update the label
            sections[-1].label = label

    for i, (kind, text, style) in enumerate(blocks):
        level = heading_level(text, style) if kind == "para" else None
        if level:
            if level == "major":
                major, minor = _label(text), ""
            else:
                minor = _label(text)
            open_section(" > ".join(x for x in (major, minor) if x))
            continue
        lead = _LEAD.match(text) if kind == "para" else None
        if lead:
            major, minor = f"{lead.group(1)}. {lead.group(2).strip()}", ""
            open_section(major)
        parts = explode_numbered(text) if kind == "para" else [text]
        for part in parts:
            sentences = split_sentences(part) if kind == "para" else [part]
            sections[-1].units += [Unit(s, i) for s in sentences]
    return [s for s in sections if s.units]


def _norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def _segment(d: list[float], lens: list[int], thr: float, min_chars: int, max_chars: int):
    """Cut where distance > thr, merge groups < min_chars, split groups > max_chars."""
    n = len(lens)
    bounds = [0] + [i + 1 for i, x in enumerate(d) if x > thr] + [n]
    groups = [[a, b] for a, b in zip(bounds, bounds[1:])]
    size = lambda g: sum(lens[g[0]:g[1]]) + (g[1] - g[0] - 1)

    while len(groups) > 1:  # merge the smallest too-small group into its closer neighbour
        small = [k for k, g in enumerate(groups) if size(g) < min_chars]
        if not small:
            break
        k = min(small, key=lambda k: size(groups[k]))
        left = d[groups[k][0] - 1] if k > 0 else np.inf
        right = d[groups[k][1] - 1] if k < len(groups) - 1 else np.inf
        j = k - 1 if left <= right else k + 1
        lo, hi = sorted((k, j))
        groups[lo:hi + 1] = [[groups[lo][0], groups[hi][1]]]

    out = []

    def split(g):  # too big -> cut at the biggest topic shift that leaves both halves usable
        a, b = g
        if size(g) <= max_chars or b - a < 2:
            out.append(g)
            return
        cand = list(range(a, b - 1))
        for floor in (min_chars, min_chars // 2, 0):  # prefer cuts that leave both halves big enough
            ok = [i for i in cand if size([a, i + 1]) >= floor and size([i + 1, b]) >= floor]
            if ok:
                break
        i = max(ok, key=lambda i: d[i])
        split([a, i + 1])
        split([i + 1, b])

    for g in groups:
        split(g)
    return out


def chunk_blocks(
    blocks: list[Block],
    embed_fn: EmbedFn,
    *,
    file_name: str = "",
    title: str = "",
    percentile: float = 90,  # semantic cut threshold, taken over the whole document
    buffer: int = 1,  # neighbours on each side that are embedded together with a unit
    min_chars: int = 300,
    max_chars: int = 1200,
    abs_floor: float | None = None,  # optional: never cut if cosine distance is below this
) -> list[Chunk]:
    title = title or next((t[:100] for k, t, _ in blocks if k == "para" and len(t) >= 15), file_name)
    sections = build_sections(blocks)
    if len(sections) > 1 and not sections[0].label and sum(len(u.text) for u in sections[0].units) < min_chars:
        sections[1].units = sections[0].units + sections[1].units  # title-only chunks are noise
        sections = sections[1:]

    # one batched embedding call for every window of every LONG section
    windows, spans = [], []
    for s in sections:
        total = sum(len(u.text) for u in s.units)
        if total <= max_chars or len(s.units) < 2:
            spans.append(None)  # short section: keep whole, no semantic cuts needed
            continue
        t = [u.text for u in s.units]
        w = [" ".join(t[max(0, i - buffer): i + buffer + 1]) for i in range(len(t))]
        spans.append((len(windows), len(windows) + len(w)))
        windows += w
    vecs = _norm(embed_fn(windows)) if windows else np.empty((0, 1))

    dists = {}
    for si, sp in enumerate(spans):
        if sp:
            v = vecs[sp[0]:sp[1]]
            dists[si] = [float(1 - v[i] @ v[i + 1]) for i in range(len(v) - 1)]
    pool = [x for d in dists.values() for x in d]
    thr = float(np.percentile(pool, percentile)) if pool else 1.0
    if abs_floor is not None:
        thr = max(thr, abs_floor)

    chunks: list[Chunk] = []
    for si, s in enumerate(sections):
        lens = [len(u.text) for u in s.units]
        groups = _segment(dists[si], lens, thr, min_chars, max_chars) if si in dists else [[0, len(s.units)]]
        for a, b in groups:
            parts = [s.units[a].text]
            for prev, u in zip(s.units[a:b - 1], s.units[a + 1:b]):
                parts.append(("\n" if u.block != prev.block else " ") + u.text)
            content = "".join(parts)
            ctx = " > ".join(x for x in (title, s.label) if x)
            chunks.append(Chunk(
                content=content,
                embed_text=f"{ctx}\n{content}" if ctx else content,
                section=s.label,
                index=len(chunks),
                metadata={"file_name": file_name, "section": s.label, "chunk_index": len(chunks),
                          "char_len": len(content), "unit_range": [a, b]},
            ))
    return chunks


# --------------------------------------------------------------------------- #
# 5. Measure instead of guessing                                               #
# --------------------------------------------------------------------------- #
def evaluate(chunks: list[Chunk], embed_fn: EmbedFn, cases: list[tuple[str, list[str]]], k: int = 3) -> float:
    """cases = [(question, [substrings that must appear in the top-k chunks]), ...]"""
    vecs = _norm(embed_fn([c.embed_text for c in chunks]))
    hits = 0
    for q, needles in cases:
        top = np.argsort(-(vecs @ _norm(embed_fn([q]))[0]))[:k]
        text = " ".join(chunks[i].content for i in top).lower()
        missing = [n for n in needles if n.lower() not in text]
        hits += not missing
        print(f"  {'OK  ' if not missing else 'MISS'} {q}" + (f"   missing: {missing}" if missing else ""))
    rate = hits / len(cases)
    print(f"  hit@{k} = {hits}/{len(cases)} = {rate:.0%}")
    return rate


DEMO_CASES = [
    ("Какви са необходимите документи?", ["заявление", "диплома", "автобиография", "лична карта", "трудова книжка"]),
    ("Каква е минималната заплата за съдебен секретар?", ["813"]),
    ("Какво символизират ръцете и косата при Фотев?", ["нежността"]),
    ("Кога е написана елегията на Дебелянов?", ["1913"]),
]


def _stub_embed(texts: list[str]) -> list[list[float]]:
    """Hashed character-trigram vectors. Lets the pipeline run offline; NOT a quality proxy for bge-m3."""
    out = []
    for t in texts:
        v = np.zeros(256)
        t = t.lower()
        for i in range(len(t) - 2):
            v[hash(t[i:i + 3]) % 256] += 1
        out.append(v)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = args[0] if args else "../documents"
    if "--stub" in sys.argv:
        embed = _stub_embed
    else:
        from llama_index.embeddings.ollama import OllamaEmbedding

        embed = OllamaEmbedding(model_name="bge-m3", base_url="http://localhost:11434").get_text_embedding_batch

    from parsers import read_blocks, SUPPORTED  # only the demo CLI needs the parsers

    all_chunks: list[Chunk] = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(SUPPORTED):
            cs = chunk_blocks(read_blocks(os.path.join(folder, name)), embed, file_name=name)
            sizes = [len(c.content) for c in cs]
            print(f"\n{name}: {len(cs)} chunks, chars min/median/max = {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
            for c in cs:
                print(f"  [{c.index:2}] {len(c.content):5} ch | {c.section[:40]:40} | {c.content[:60]!r}")
            all_chunks += cs
    print("\nRetrieval check over all documents:")
    evaluate(all_chunks, embed, DEMO_CASES)