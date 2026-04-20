"""
Knowledge base for Phase 2.

Loads all HTML files from phase2_data/, parses the tables with BeautifulSoup,
creates one KnowledgeChunk per (category, service, hmo, tier) combination,
then embeds every chunk using Azure OpenAI ADA-002.

Retrieval:
  retrieve(query, hmo, tier, top_k) — returns the top_k most semantically
  relevant chunks that belong to the requested HMO and tier.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from openai import AzureOpenAI, APIError, RateLimitError, APIConnectionError

from phase2.backend.models import KnowledgeChunk

logger = logging.getLogger(__name__)

# ── Column → HMO mapping (by position in every table) ───────────────────────
# Column 0 = service name, 1 = מכבי, 2 = מאוחדת, 3 = כללית
HMO_COLUMNS = ["מכבי", "מאוחדת", "כללית"]

# ── Tier keyword mapping ──────────────────────────────────────────────────────
TIER_KEYS = {
    "זהב": "זהב",
    "כסף": "כסף",
    "ארד": "ארד",
}

# ── File → category name ─────────────────────────────────────────────────────
FILE_CATEGORIES = {
    "alternative_services.html":    "רפואה משלימה",
    "communication_clinic_services.html": "מרפאות תקשורת",
    "dentel_services.html":         "מרפאות שיניים",
    "optometry_services.html":      "אופטומטריה",
    "pragrency_services.html":      "הריון",
    "workshops_services.html":      "סדנאות בריאות",
}


def _openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with ADA-002.  Retries on rate-limit errors."""
    client   = _openai_client()
    model    = os.environ.get("AZURE_OPENAI_DEPLOYMENT_ADA002", "text-embedding-ada-002")
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except RateLimitError as exc:
            import time
            wait = 2 ** attempt
            logger.warning("Rate limit hit (attempt %d/%d). Retrying in %ds: %s", attempt, max_retries, wait, exc)
            time.sleep(wait)
        except (APIConnectionError, APIError) as exc:
            logger.error("Embedding API error: %s", exc)
            raise

    raise RuntimeError("Embedding failed after max retries due to rate limiting.")


def _parse_tier_cell(cell_html: str) -> dict[str, str]:
    """
    Parse one table cell that contains tier-specific text like:
        <strong>זהב:</strong> 70% הנחה, עד 20 טיפולים בשנה<br>
        <strong>כסף:</strong> 50% הנחה, עד 12 טיפולים בשנה<br>
        ...

    Returns {"זהב": "70% הנחה, עד 20 טיפולים בשנה", "כסף": ..., "ארד": ...}
    """
    soup   = BeautifulSoup(cell_html, "html.parser")
    result = {}

    # Replace <br> with newline so we can split cleanly
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.get_text(separator="\n")

    for tier_he in TIER_KEYS:
        # Look for lines starting with the tier name (possibly with colon)
        pattern = rf"(?m)^{re.escape(tier_he)}\s*[:\u05be]?\s*(.+?)(?=(?:זהב|כסף|ארד)\s*[:\u05be]|$)"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            value = m.group(1).strip().replace("\n", " ").strip()
            if value:
                result[tier_he] = value

    return result


def _parse_html_file(html_path: Path) -> list[tuple[str, str, str, str]]:
    """
    Parse one HTML knowledge-base file.
    Returns list of (category, service_name, hmo, tier, benefit_text) tuples
    as (category, service, hmo, tier_text) where tier_text already includes tier label.
    """
    category = FILE_CATEGORIES.get(html_path.name, html_path.stem)

    with open(html_path, encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "html.parser")

    table = soup.find("table")
    if not table:
        logger.warning("No table found in %s", html_path.name)
        return []

    rows = table.find_all("tr")
    records: list[tuple[str, str, str, str]] = []

    for row in rows[1:]:   # skip header row
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        service_name = cells[0].get_text(strip=True)

        for col_idx, hmo in enumerate(HMO_COLUMNS, start=1):
            if col_idx >= len(cells):
                continue
            cell_html = str(cells[col_idx])
            tier_map  = _parse_tier_cell(cell_html)

            for tier_he, benefit in tier_map.items():
                records.append((category, service_name, hmo, tier_he, benefit))

    # ── Extract contact info sections below the table ─────────────────────────
    # Collect all <li> text after the table, grouped by HMO name
    contact_by_hmo: dict[str, list[str]] = {}
    for li in table.find_all_next("li"):
        text = li.get_text(separator=" ", strip=True)
        for hmo in HMO_COLUMNS:
            if text.startswith(hmo):
                contact_by_hmo.setdefault(hmo, []).append(text)

    # Create one contact chunk per HMO × tier so retrieval finds it
    for hmo, lines in contact_by_hmo.items():
        contact_text = "\n".join(lines)
        for tier_he in TIER_KEYS:
            records.append((category, "יצירת קשר ופרטים נוספים", hmo, tier_he, contact_text))

    return records


class KnowledgeBase:
    """
    In-memory vector store of KnowledgeChunks.

    Built once at startup and reused across all requests (stateless retrieval).
    """

    def __init__(self) -> None:
        self._chunks: list[KnowledgeChunk] = []
        self._matrix: np.ndarray | None = None  # shape (N, 1536)

    def load(self, data_dir: str | Path) -> None:
        """Parse all HTML files and embed the resulting chunks."""
        data_dir = Path(data_dir)
        html_files = [data_dir / name for name in FILE_CATEGORIES if (data_dir / name).exists()]

        if not html_files:
            raise FileNotFoundError(
                f"No knowledge-base HTML files found in '{data_dir}'. "
                "Ensure PHASE2_DATA_PATH points to the phase2_data folder."
            )

        logger.info("Loading knowledge base from %d HTML files...", len(html_files))

        all_records: list[tuple] = []
        for path in html_files:
            records = _parse_html_file(path)
            logger.info("  %s → %d records", path.name, len(records))
            all_records.extend(records)

        if not all_records:
            raise ValueError("Knowledge base parsing produced zero records. Check HTML structure.")

        # Build chunks
        chunks: list[KnowledgeChunk] = []
        texts: list[str] = []

        for category, service, hmo, tier, benefit in all_records:
            content = (
                f"קטגוריה: {category}\n"
                f"שירות: {service}\n"
                f"קופה: {hmo}\n"
                f"מסלול: {tier}\n"
                f"הטבה: {benefit}"
            )
            chunks.append(KnowledgeChunk(
                category=category,
                service=service,
                hmo=hmo,
                tier=tier,
                content=content,
            ))
            texts.append(content)

        # Embed in batches of 100 (ADA-002 limit)
        logger.info("Embedding %d chunks with ADA-002...", len(texts))
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = _embed_texts(batch)
            all_embeddings.extend(embeddings)
            logger.info("  Embedded %d / %d", min(i + batch_size, len(texts)), len(texts))

        for chunk, emb in zip(chunks, all_embeddings):
            chunk.embedding = emb

        self._chunks = chunks
        self._matrix = np.array(all_embeddings, dtype=np.float32)
        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        self._matrix = self._matrix / np.maximum(norms, 1e-9)

        logger.info("Knowledge base ready: %d chunks embedded.", len(self._chunks))

    def retrieve(
        self,
        query: str,
        hmo: str,
        tier: str,
        top_k: int = 5,
    ) -> list[KnowledgeChunk]:
        """
        Embed the query, filter chunks by HMO + tier, rank by cosine similarity,
        return the top_k most relevant chunks.
        """
        if self._matrix is None or not self._chunks:
            raise RuntimeError("KnowledgeBase not loaded. Call load() first.")

        # Embed the query
        q_emb = np.array(_embed_texts([query])[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 1e-9:
            q_emb /= q_norm

        # Filter indices by HMO and tier
        indices = [
            i for i, c in enumerate(self._chunks)
            if c.hmo == hmo and c.tier == tier
        ]

        if not indices:
            logger.warning(
                "No chunks found for hmo='%s' tier='%s'. Returning unfiltered top results.", hmo, tier
            )
            indices = list(range(len(self._chunks)))

        sub_matrix = self._matrix[indices]                     # (M, 1536)
        scores     = sub_matrix @ q_emb                       # (M,)
        top_local  = np.argsort(scores)[::-1][:top_k]         # descending

        return [self._chunks[indices[i]] for i in top_local]

    def get_contact_chunk(self, hmo: str, tier: str, category: str) -> "KnowledgeChunk | None":
        """Return the contact/registration chunk for a specific HMO, tier, and category."""
        for c in self._chunks:
            if c.hmo == hmo and c.tier == tier and c.category == category \
                    and c.service == "יצירת קשר ופרטים נוספים":
                return c
        return None

    def get_full_context_for_user(self, hmo: str, tier: str) -> str:
        """
        Return ALL content relevant to a specific HMO+tier as a single block.
        Used when no specific query is given (e.g. for broad questions).
        """
        relevant = [c for c in self._chunks if c.hmo == hmo and c.tier == tier]
        return "\n\n---\n\n".join(c.content for c in relevant)


# ── Module-level singleton ────────────────────────────────────────────────────
_kb: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """Return the singleton, initialising it lazily on first call."""
    global _kb
    if _kb is None:
        data_dir = Path(os.environ.get("PHASE2_DATA_PATH",
                        Path(__file__).resolve().parent.parent.parent / "phase2_data"))
        logger.info("Lazy-loading knowledge base from: %s", data_dir)
        _kb = KnowledgeBase()
        _kb.load(data_dir)
    return _kb


def init_knowledge_base(data_dir: str | Path) -> KnowledgeBase:
    """Create and load the singleton knowledge base. Safe to call once at startup."""
    global _kb
    _kb = KnowledgeBase()
    _kb.load(data_dir)
    return _kb
