"""Text chunking with overlap.

Splits on paragraph/sentence boundaries where possible, then packs into windows of
roughly ``chunk_size`` characters with ``overlap`` carried between chunks so context
isn't lost at boundaries.
"""

from __future__ import annotations

import re


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Prefer splitting on blank lines, then single newlines, then sentences.
    units = _split_units(text)
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(current) + len(unit) + 1 <= chunk_size:
            current = f"{current}\n{unit}" if current else unit
        else:
            if current:
                chunks.append(current)
            # Start the next chunk with tail overlap of the previous one.
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n{unit}" if tail else unit
            # A single oversized unit is hard-split.
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap :]
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def _split_units(text: str) -> list[str]:
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= 1200:
            units.append(p.strip())
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())
    return units
