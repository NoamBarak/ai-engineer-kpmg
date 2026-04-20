"""
Azure Document Intelligence wrapper for OCR extraction.
Uses prebuilt-layout model with rich spatial output for Form 283.

Key improvements over plain text output:
- Each OCR line is prefixed with its normalised vertical position [y:0.NNN]
  so the LLM can reason about spatial proximity for offset handwriting.
- An explicit "=== Checkbox States ===" section maps every selection mark to
  its nearest text label via physical coordinates.
- Inline :selected:/:unselected: tokens are preserved as a fallback signal.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError

# Raw API string — avoids SDK-version enum serialisation bugs ("SELECTION_MARKS" vs "selectionMarks")
_SELECTION_MARKS_FEATURE = "selectionMarks"

logger = logging.getLogger(__name__)


# ── Azure client ───────────────────────────────────────────────────────────────

def _build_client() -> DocumentIntelligenceClient:
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
    if not endpoint or not key:
        raise EnvironmentError(
            "Missing AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or AZURE_DOCUMENT_INTELLIGENCE_KEY"
        )
    return DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_text_from_bytes(file_bytes: bytes, content_type: str = "application/pdf") -> str:
    """
    Send raw bytes to Azure Document Intelligence (prebuilt-layout) and return a
    rich structured text string for the LLM:
      - Each line prefixed with normalised y-coordinate [y:0.NNN]
      - An explicit 'Checkbox States' section listing SELECTED/NOT SELECTED labels
      - Table data in pipe-delimited rows
    """
    client = _build_client()

    try:
        logger.info("Sending document to Azure Document Intelligence (layout model)...")
        try:
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                file_bytes,
                content_type=content_type,
                features=[_SELECTION_MARKS_FEATURE],
            )
        except HttpResponseError as feat_exc:
            logger.warning(
                "selectionMarks feature rejected by endpoint (%s). "
                "Retrying without it — checkboxes fall back to inline :selected: tokens.",
                feat_exc.message,
            )
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                file_bytes,
                content_type=content_type,
            )
        result = poller.result()
        logger.info("OCR completed. Pages: %d", len(result.pages) if result.pages else 0)
    except HttpResponseError as exc:
        logger.error("Azure Document Intelligence HTTP error: %s", exc)
        raise RuntimeError(f"OCR request failed: {exc.message}") from exc
    except ServiceRequestError as exc:
        logger.error("Network error reaching Document Intelligence: %s", exc)
        raise RuntimeError(f"Network error during OCR: {exc}") from exc

    layout = _result_to_layout_dict(result)
    return _layout_to_rich_text(layout)


def extract_text_from_file(file_path: str | Path) -> str:
    """Convenience wrapper: read a local file then call extract_text_from_bytes."""
    path = Path(file_path)
    content_type_map = {
        ".pdf":  "application/pdf",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".tiff": "image/tiff",
        ".bmp":  "image/bmp",
    }
    ct = content_type_map.get(path.suffix.lower(), "application/octet-stream")
    with open(path, "rb") as fh:
        return extract_text_from_bytes(fh.read(), ct)


# ── Spatial helpers ────────────────────────────────────────────────────────────

def _poly_to_floats(polygon: Any) -> list[float]:
    """Convert any polygon representation to a flat list of floats."""
    if not polygon:
        return []
    return [float(p) for p in polygon]


def _center_from_polygon(polygon: list[float]) -> tuple[float, float]:
    """Return the centroid (cx, cy) of a polygon given as flat [x0,y0,x1,y1,…]."""
    if not polygon:
        return 0.0, 0.0
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _cluster_marks_by_row(marks: list[dict], y_tol: float = 0.25) -> list[list[dict]]:
    """
    Group selection marks into horizontal rows using y-coordinate proximity.
    y_tol: max vertical distance (inches) for marks to be considered on the same row.
    """
    if not marks:
        return []
    by_y = sorted(marks, key=lambda m: m["center_y"])
    rows: list[list[dict]] = [[by_y[0]]]
    for m in by_y[1:]:
        avg_y = sum(r["center_y"] for r in rows[-1]) / len(rows[-1])
        if abs(m["center_y"] - avg_y) <= y_tol:
            rows[-1].append(m)
        else:
            rows.append([m])
    return rows


def _label_for_mark(
    mark: dict,
    row_marks: list[dict],
    words: list[dict],
    y_tol: float = 0.25,   # Increased to catch messy handwriting
    x_range: float = 1.5,  # Increased to catch spaced-out labels
) -> str:
    """
    Find the option label for a selection mark using spatial proximity.
    Safely ignores the marks themselves and finds the nearest RTL text.
    """
    # CRITICAL: Skip all checkmark variants so they aren't selected as their own labels
    _skip = {
        ":selected:", ":unselected:", "",
        "☑", "☐", "v", "V", "x", "X", "✓", "✔",
        "[", "]", "(", ")"
    }

    avg_row_y = sum(m["center_y"] for m in row_marks) / len(row_marks)

    candidates: list[tuple[int, float, float, str]] = []
    for w in words:
        content = w["content"].strip()
        if not content or content in _skip:
            continue

        dy = abs(w["center_y"] - avg_row_y)
        if dy > y_tol:
            continue

        dx = w["center_x"] - mark["center_x"]
        dist = abs(dx)

        if dist < 0.04 or dist > x_range:
            continue

        # Prefer RIGHT side (label is to the right of checkbox in RTL forms)
        on_preferred_side = dx > 0
        priority = 0 if on_preferred_side else 1

        candidates.append((priority, dist, w["center_x"], content))

    if not candidates:
        return "(unknown)"

    # Sort by priority, then distance
    candidates.sort(key=lambda t: (t[0], t[1]))
    nearest_priority, nearest_dist, _, _ = candidates[0]

    # Cluster words within 0.5" of the nearest to form multi-word labels
    cluster = [
        (priority, dist, cx, word)
        for priority, dist, cx, word in candidates
        if priority == nearest_priority and dist <= nearest_dist + 0.50
    ]

    # Sort cluster descending (RTL reading order)
    cluster.sort(key=lambda t: -t[2])

    words_in_label = [word for _, _, _, word in cluster[:5]]
    label = " ".join(words_in_label)

    # Clean up trailing colons if it accidentally caught a header prefix
    if label.endswith(":"):
        label = label[:-1].strip()

    return label if label else "(unknown)"


# ── Layout extraction ──────────────────────────────────────────────────────────

def _result_to_layout_dict(result: Any) -> dict[str, Any]:
    """Convert Azure DI AnalyzeResult into a plain dict preserving spatial metadata."""
    doc: dict[str, Any] = {
        "pages":   [],
        "tables":  [],
        # Full document text produced by Azure DI's own reading-order algorithm.
        # Selection mark tokens (:selected: / :unselected:) appear inline here
        # at exactly the position Azure determined — used for context-window extraction.
        "content": result.content or "",
    }

    for page in result.pages or []:
        page_obj: dict[str, Any] = {
            "page_number": page.page_number,
            "width":  float(page.width)  if page.width  is not None else None,
            "height": float(page.height) if page.height is not None else None,
            "lines":            [],
            "words":            [],
            "selection_marks":  [],
        }

        for line in page.lines or []:
            poly = _poly_to_floats(getattr(line, "polygon", None))
            cx, cy = _center_from_polygon(poly)
            page_obj["lines"].append({
                "content":  (line.content or "").strip(),
                "center_x": cx,
                "center_y": cy,
            })

        for word in getattr(page, "words", []) or []:
            poly = _poly_to_floats(getattr(word, "polygon", None))
            cx, cy = _center_from_polygon(poly)
            content = (word.content or "").strip()
            
            page_obj["words"].append({
                "content":  content,
                "center_x": cx,
                "center_y": cy,
            })

            # Intercept tiny 'v's, 'x's, and Unicode boxes and force them into selection_marks
            if content in ("☑", "v", "V", "x", "X", "✓", "✔"):
                page_obj["selection_marks"].append({
                    "state": "selected",
                    "center_x": cx,
                    "center_y": cy,
                    "spans": []
                })
            elif content == "☐":
                page_obj["selection_marks"].append({
                    "state": "unselected",
                    "center_x": cx,
                    "center_y": cy,
                    "spans": []
                })

        for mark in getattr(page, "selection_marks", []) or []:
            poly = _poly_to_floats(getattr(mark, "polygon", None))
            cx, cy = _center_from_polygon(poly)
            # Collect span offsets into result.content (the authoritative source).
            spans = [
                {"offset": int(s.offset), "length": int(s.length)}
                for s in (getattr(mark, "spans", None) or [])
            ]
            page_obj["selection_marks"].append({
                "state":    str(getattr(mark, "state", "") or "").lower(),
                "center_x": cx,
                "center_y": cy,
                "spans":    spans,
            })

        doc["pages"].append(page_obj)

    for table in result.tables or []:
        table_obj: dict[str, Any] = {
            "row_count":    table.row_count,
            "column_count": table.column_count,
            "cells": [],
        }
        for cell in table.cells:
            table_obj["cells"].append({
                "row_index":    cell.row_index,
                "column_index": cell.column_index,
                "content":      (cell.content or "").strip(),
            })
        doc["tables"].append(table_obj)

    return doc


# ── Rich text renderer ─────────────────────────────────────────────────────────

def _layout_to_rich_text(layout: dict[str, Any]) -> str:
    """
    Render the layout dict as a rich text string for the LLM.

    Per-page output:
      === Page N ===
      [y:0.NNN x:0.NNN] <line text, with inline :selected:/:unselected: tokens>
      ...

      === Checkbox States ===
        SELECTED:     "<±80-char context snippet from Azure DI reading order>" [y:0.NNN]
        NOT SELECTED: "<±80-char context snippet from Azure DI reading order>" [y:0.NNN]
        ...

    The context snippet is extracted by taking the span offset Azure DI provides for
    each selection mark and slicing result.content at ±80 chars around that offset.
    This places the :selected:/:unselected: token inline with the surrounding Hebrew
    label text exactly as Azure's reading-order algorithm determined — no coordinate
    maths required.  Falls back to spatial word matching if spans are unavailable.

    Followed by table data.
    """
    parts: list[str] = []

    for page in layout.get("pages", []):
        page_h = page.get("height") or 1.0
        words  = page.get("words", [])

        # ── Text lines with y-coordinate prefix ───────────────────────────────
        page_lines: list[str] = []
        page_w = page.get("width") or 1.0
        for line in page.get("lines", []):
            # Keep inline :selected:/:unselected: tokens — they serve as fallback
            # when the Checkbox States section cannot resolve a label spatially.
            content = " ".join(line.get("content", "").strip().split())
            if not content:
                continue
            y_norm = line.get("center_y", 0.0) / page_h
            x_norm = line.get("center_x", 0.0) / page_w
            page_lines.append(f"[y:{y_norm:.3f} x:{x_norm:.3f}] {content}")

        parts.append(f"=== Page {page['page_number']} ===\n" + "\n".join(page_lines))

        # ── Explicit checkbox states ───────────────────────────────────────────
        marks = page.get("selection_marks", [])
        if marks:
            mark_lines: list[str] = []
            # Group marks into rows so header exclusion works correctly.
            # (section headers sit to the RIGHT of all marks in their row;
            #  without grouping we cannot distinguish header from label words.)
            rows = _cluster_marks_by_row(marks)
            for row in rows:
                # Sort each row right-to-left (descending x) = RTL reading order
                for mark in sorted(row, key=lambda m: -m["center_x"]):
                    is_selected = mark.get("state", "") == "selected"
                    label = _label_for_mark(mark, row, words)
                    y_norm = mark["center_y"] / page_h
                    status = "SELECTED    " if is_selected else "NOT SELECTED"
                    mark_lines.append(f"  {status}: \"{label}\" [y:{y_norm:.3f}]")
            parts.append("=== Checkbox States ===\n" + "\n".join(mark_lines))

    # ── Tables ─────────────────────────────────────────────────────────────────
    tables = layout.get("tables", [])
    if tables:
        table_texts: list[str] = []
        for t_idx, table in enumerate(tables, start=1):
            rows: dict[int, dict[int, str]] = {}
            for cell in table.get("cells", []):
                rows.setdefault(cell["row_index"], {})[cell["column_index"]] = cell.get("content", "")
            row_strs: list[str] = []
            for row_idx in sorted(rows):
                row = rows[row_idx]
                max_col = max(row) if row else -1
                row_strs.append(" | ".join(row.get(c, "") for c in range(max_col + 1)))
            table_texts.append(f"[Table {t_idx}]\n" + "\n".join(row_strs))
        parts.append("=== Extracted Tables ===\n" + "\n\n".join(table_texts))

    return "\n\n".join(parts)