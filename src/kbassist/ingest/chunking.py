"""Structure-aware chunking.

* Markdown is split on headings (ignoring `#` lines inside code fences) and each
  chunk carries its heading breadcrumb, so "Fine tuning the configuration" is
  retrievable as "Timeouts > Fine tuning the configuration".
* Python is split with `ast` into module header, top-level functions and classes;
  large classes are split per method with the class signature kept as context.
* Anything else falls back to line windows.
Oversized pieces are split on blank lines / line boundaries with overlap.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass

from kbassist.models import Chunk

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class ChunkParams:
    max_chars: int = 1800
    min_chars: int = 200
    overlap_chars: int = 200


@dataclass
class _Piece:
    title: str
    start: int  # 1-based inclusive
    end: int
    text: str


def _chunk_id(source: str, path: str, start: int, end: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{path}:{start}:{end}:{text}".encode()).hexdigest()
    return h[:16]


def _split_oversized(piece: _Piece, p: ChunkParams) -> list[_Piece]:
    if len(piece.text) <= p.max_chars:
        return [piece]
    lines = piece.text.split("\n")
    out: list[_Piece] = []
    buf: list[str] = []
    buf_start = 0
    size = 0
    for i, line in enumerate(lines):
        # Prefer to break on a blank line once we are past the soft limit.
        if buf and size + len(line) > p.max_chars:
            out.append(_Piece(piece.title, piece.start + buf_start, piece.start + i - 1, "\n".join(buf)))
            # carry overlap
            carry: list[str] = []
            carry_size = 0
            for prev in reversed(buf):
                if carry_size + len(prev) > p.overlap_chars:
                    break
                carry.insert(0, prev)
                carry_size += len(prev) + 1
            buf = carry
            buf_start = i - len(carry)
            size = carry_size
        buf.append(line)
        size += len(line) + 1
    if buf:
        out.append(_Piece(piece.title, piece.start + buf_start, piece.start + len(lines) - 1, "\n".join(buf)))
    return out


def _merge_small(pieces: list[_Piece], p: ChunkParams) -> list[_Piece]:
    """Fold tiny sections (e.g. a heading with one sentence) into the next one."""
    merged: list[_Piece] = []
    pending: _Piece | None = None
    for piece in pieces:
        if pending is not None:
            piece = _Piece(
                pending.title if len(piece.text) < p.min_chars else piece.title,
                pending.start,
                piece.end,
                pending.text + "\n" + piece.text,
            )
            pending = None
        if len(piece.text) < p.min_chars:
            pending = piece
        else:
            merged.append(piece)
    if pending is not None:
        if merged and len(merged[-1].text) + len(pending.text) <= p.max_chars:
            last = merged.pop()
            merged.append(_Piece(last.title, last.start, pending.end, last.text + "\n" + pending.text))
        else:
            merged.append(pending)
    return merged


def chunk_markdown(path: str, text: str, p: ChunkParams) -> list[_Piece]:
    lines = text.split("\n")
    stack: list[tuple[int, str]] = []
    pieces: list[_Piece] = []
    start = 0
    in_fence = False

    def flush(end_idx: int) -> None:
        body = "\n".join(lines[start:end_idx]).rstrip()
        if body.strip():
            crumb = " > ".join(t for _, t in stack) or path
            pieces.append(_Piece(crumb, start + 1, end_idx, body))

    for i, line in enumerate(lines):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        m = None if in_fence else HEADING.match(line)
        if m:
            flush(i)
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).replace("`", "").strip()))
            start = i
    flush(len(lines))

    pieces = _merge_small(pieces, p)
    return [s for piece in pieces for s in _split_oversized(piece, p)]


def chunk_python(path: str, text: str, p: ChunkParams) -> list[_Piece]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_lines(path, text, p)
    lines = text.split("\n")
    module = path.removesuffix(".py").replace("/", ".")

    def seg(node: ast.AST) -> tuple[int, int]:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        return start, node.end_lineno or node.lineno

    def src(a: int, b: int) -> str:
        return "\n".join(lines[a - 1 : b])

    pieces: list[_Piece] = []
    header_end = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            break
        header_end = node.end_lineno or header_end
    if header_end:
        pieces.append(_Piece(f"{module} (module header)", 1, header_end, src(1, header_end)))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a, b = seg(node)
            pieces.append(_Piece(f"{module}.{node.name}", a, b, src(a, b)))
        elif isinstance(node, ast.ClassDef):
            a, b = seg(node)
            body = src(a, b)
            if len(body) <= p.max_chars:
                pieces.append(_Piece(f"{module}.{node.name}", a, b, body))
                continue
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            first = seg(methods[0])[0] if methods else b + 1
            class_head = src(a, first - 1).rstrip()
            pieces.append(_Piece(f"{module}.{node.name}", a, first - 1, class_head))
            signature = lines[node.lineno - 1].strip()
            for m in methods:
                ma, mb = seg(m)
                pieces.append(
                    _Piece(f"{module}.{node.name}.{m.name}", ma, mb, f"{signature}\n    ...\n{src(ma, mb)}")
                )
        elif (node.lineno or 0) > header_end:
            # module-level statements after the first def (constants, __all__, ...)
            a, b = seg(node)
            pieces.append(_Piece(f"{module} (module level)", a, b, src(a, b)))

    pieces = _merge_adjacent_module_level(pieces, p)
    return [s for piece in pieces for s in _split_oversized(piece, p)]


def _merge_adjacent_module_level(pieces: list[_Piece], p: ChunkParams) -> list[_Piece]:
    out: list[_Piece] = []
    for piece in pieces:
        if (
            out
            and piece.title.endswith("(module level)")
            and out[-1].title.endswith("(module level)")
            and len(out[-1].text) + len(piece.text) < p.max_chars
        ):
            last = out.pop()
            piece = _Piece(last.title, last.start, piece.end, last.text + "\n" + piece.text)
        out.append(piece)
    return out


def chunk_lines(path: str, text: str, p: ChunkParams) -> list[_Piece]:
    n = text.count("\n") + 1
    return _split_oversized(_Piece(path, 1, n, text), p)


def chunk_file(source: str, path: str, text: str, p: ChunkParams) -> list[Chunk]:
    if path.endswith(".md"):
        kind, pieces = "doc", chunk_markdown(path, text, p)
    elif path.endswith(".py"):
        kind, pieces = "code", chunk_python(path, text, p)
    else:
        kind, pieces = "config", chunk_lines(path, text, p)
    return [
        Chunk(
            id=_chunk_id(source, path, pc.start, pc.end, pc.text),
            source=source,
            path=path,
            kind=kind,
            title=pc.title,
            start_line=pc.start,
            end_line=pc.end,
            text=pc.text,
        )
        for pc in pieces
        if pc.text.strip()
    ]
