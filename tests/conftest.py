"""
Shared fixtures and path bootstrap for Phase 1 & Phase 2 tests.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../Home-Assignment-GenAI-KPMG-.../
FILES_DIR    = PROJECT_ROOT / "files"          # .../files/
PHASE1_DATA  = PROJECT_ROOT / "phase1_data"           # .../phase1_data/
PHASE2_DATA  = PROJECT_ROOT / "phase2_data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests that call real Azure services (OCR, GPT-4o, ADA-002).",
    )


@pytest.fixture(scope="session")
def run_integration(request):
    return request.config.getoption("--run-integration")


# ── Phase 1 data fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def correct_jsons() -> dict[str, dict]:
    """Load all three ground-truth JSONs, keyed by '1', '2', '3'."""
    result = {}
    for i in (1, 2, 3):
        with open(FILES_DIR / f"ex{i}_correct.json", encoding="utf-8") as fh:
            result[str(i)] = json.load(fh)
    return result


@pytest.fixture(scope="session")
def pdf_paths() -> dict[str, Path]:
    """Return paths to the three PDF form files."""
    return {str(i): PHASE1_DATA / f"283_ex{i}.pdf" for i in (1, 2, 3)}


@pytest.fixture(scope="session")
def wrong_jsons() -> dict[str, dict]:
    """Load the three pre-computed wrong JSONs, keyed by '1', '2', '3'."""
    result = {}
    for i in (1, 2, 3):
        path = FILES_DIR / f"ex{i}_wrong.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                result[str(i)] = json.load(fh)
    return result


@pytest.fixture(scope="session")
def valid_extraction_base(correct_jsons) -> dict:
    """
    A structurally-valid extraction dict suitable for testing validate_extraction()
    without triggering any warnings or errors.

    All personal fields (name, ID, address, phone) are taken verbatim from
    ex2_correct.json so no user data is invented.  The only adjustment is
    formReceiptDateAtClinic, which is pushed one year past formFillingDate so
    all chronological constraints are satisfied.
    """
    base = copy.deepcopy(correct_jsons["2"])
    filling_year = int(base["formFillingDate"]["year"])
    base["formReceiptDateAtClinic"] = {
        "day":   base["formFillingDate"]["day"],
        "month": base["formFillingDate"]["month"],
        "year":  str(filling_year + 1),
    }
    return base


# ── Phase 2 data fixtures ──────────────────────────────────────────────────────

def _parse_conversation(path: Path) -> list[dict[str, str]]:
    """
    Parse a conversation .txt file into turns: [{"role": "user"|"assistant", "content": ...}].
    Each block begins with 'user:' or 'assistant:' and may span multiple lines.
    """
    turns: list[dict[str, str]] = []
    role: str | None = None
    lines: list[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("assistant:"):
            if role:
                turns.append({"role": role, "content": "\n".join(lines).strip()})
            role, lines = "assistant", [raw[len("assistant:"):].strip()]
        elif raw.startswith("user:"):
            if role:
                turns.append({"role": role, "content": "\n".join(lines).strip()})
            role, lines = "user", [raw[len("user:"):].strip()]
        elif role is not None:
            lines.append(raw)

    if role:
        turns.append({"role": role, "content": "\n".join(lines).strip()})

    return turns


@pytest.fixture(scope="session")
def phase2_conversations() -> dict[str, list[dict[str, str]]]:
    """Return all five Phase 2 example conversations."""
    mapping = {
        "phase2_ex1": "ex1",
        "phase2_ex2": "ex2",
        "phase2_ex3": "ex3",
        "phase2_ex4": "ex4",
        "phase2-conv": "conv",
    }
    result = {}
    for stem, key in mapping.items():
        path = FILES_DIR / f"{stem}.txt"
        if path.exists():
            result[key] = _parse_conversation(path)
    return result


# ── Profile parsing helpers ────────────────────────────────────────────────────

_HMO_MAP  = {"Maccabi": "מכבי", "Meuhedet": "מאוחדת", "Clalit": "כללית",
             "clalit": "כללית", "maccabi": "מכבי", "meuhedet": "מאוחדת"}
_TIER_MAP = {"Gold": "זהב", "Silver": "כסף", "Bronze": "ארד",
             "gold": "זהב", "silver": "כסף", "bronze": "ארד"}
_GENDER_MAP = {"Male": "זכר", "Female": "נקבה", "male": "זכר", "female": "נקבה"}


def _parse_profile(text: str) -> dict:
    """
    Extract profile fields from a conversation summary block.
    Handles both Hebrew-label and English-label summary formats.
    Returns a dict with keys matching UserProfile fields.
    """
    import re
    p: dict = {}

    # ── Name ──────────────────────────────────────────────────────────────────
    # "שם פרטי ושם משפחה: עמית לוי"  (ex4 format)
    m = re.search(r'שם פרטי ושם משפחה\s*[:\u05BE]\s*\*{0,2}([\u0590-\u05FF]+)\s+([\u0590-\u05FF]+)', text)
    if m:
        p["firstName"], p["lastName"] = m.group(1), m.group(2)
    else:
        # "שם מלא: רויטל לוי"  (ex2 format — full name on one line)
        m = re.search(r'שם מלא\s*[:\u05BE]\s*\*{0,2}([\u0590-\u05FF]+)\s+([\u0590-\u05FF]+)', text)
        if m:
            p["firstName"], p["lastName"] = m.group(1), m.group(2)
        else:
            # "שם פרטי: יוסי" / "First Name: Karlos"  (ex1 / ex3 format — separate lines)
            m = re.search(r'(?:שם פרטי|First Name)\s*[:\u05BE]\s*\*{0,2}(\S+)', text, re.IGNORECASE)
            if m: p["firstName"] = m.group(1).strip("*")
            m = re.search(r'(?:שם משפחה|Last Name)\s*[:\u05BE]\s*\*{0,2}(\S+)', text, re.IGNORECASE)
            if m: p["lastName"] = m.group(1).strip("*")

    # ── ID number (first 9-digit sequence after an ID label) ──────────────────
    m = re.search(r'(?:זהות|ת\.ז\.|ID Number|id)\s*[:\u05BE]?\s*\*{0,2}(\d{9})\b', text, re.IGNORECASE)
    if m: p["idNumber"] = m.group(1)

    # ── Gender ────────────────────────────────────────────────────────────────
    m = re.search(r'(?:מין|Gender)\s*[:\u05BE]\s*\*{0,2}(\S+)', text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip("*")
        p["gender"] = _GENDER_MAP.get(raw, raw)

    # ── Age ───────────────────────────────────────────────────────────────────
    m = re.search(r'(?:גיל|Age)\s*[:\u05BE]\s*\*{0,2}(\d{1,3})\b', text, re.IGNORECASE)
    if m: p["age"] = int(m.group(1))

    # ── HMO ───────────────────────────────────────────────────────────────────
    m = re.search(r'(?:קופת[^:]*|HMO)\s*[:\u05BE]\s*\*{0,2}(\S+)', text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip("*|").strip()
        p["hmo"] = _HMO_MAP.get(raw, raw)

    # ── HMO card number (9-digit, distinct from idNumber) ─────────────────────
    m = re.search(r'(?:כרטיס קופת חולים|HMO Card Number)\s*[:\u05BE]\s*\*{0,2}(\d{9})\b', text, re.IGNORECASE)
    if m: p["hmoCardNumber"] = m.group(1)

    # ── Tier ──────────────────────────────────────────────────────────────────
    m = re.search(r'(?:מסלול[^:]*|Insurance Tier)\s*[:\u05BE]\s*\*{0,2}(\S+)', text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip("*")
        p["tier"] = _TIER_MAP.get(raw, raw)

    return p


@pytest.fixture(scope="session")
def phase2_profiles(phase2_conversations) -> dict[str, dict]:
    """
    For each conversation, parse the LAST assistant summary turn and extract
    the confirmed user profile. Returns dict keyed by example id.
    """
    profiles: dict[str, dict] = {}
    for key, turns in phase2_conversations.items():
        # Find the last assistant turn that contains a 9-digit number (profile summary)
        for turn in reversed(turns):
            if turn["role"] == "assistant":
                parsed = _parse_profile(turn["content"])
                if "idNumber" in parsed and "hmo" in parsed:
                    profiles[key] = parsed
                    break
    return profiles


@pytest.fixture(scope="session")
def phase2_invalid_inputs(phase2_conversations) -> dict[str, list[str]]:
    """
    Extract user-provided values that were rejected by the bot (invalid ID, age, etc.).
    Returns dict: {"invalid_ids": [...], "invalid_ages": [...]}
    """
    import re
    invalid_ids:  list[str] = []
    invalid_ages: list[str] = []

    for turns in phase2_conversations.values():
        for i, turn in enumerate(turns):
            if turn["role"] != "user":
                continue
            # Check if the NEXT assistant turn contains a rejection about this value
            next_assistant = next(
                (turns[j]["content"] for j in range(i + 1, len(turns))
                 if turns[j]["role"] == "assistant"),
                ""
            )
            rejection_keywords = ["תקין", "תקף", "valid", "9 ספרות", "9 digits",
                                   "0 עד 120", "0-120", "invalid", "שגוי"]
            if any(kw in next_assistant for kw in rejection_keywords):
                digits = re.findall(r'\d+', turn["content"])
                for d in digits:
                    if len(d) > 9 or len(d) < 9:
                        if len(d) >= 6:   # looks like an ID attempt
                            invalid_ids.append(d)
                    if 1 < len(d) <= 3:
                        val = int(d)
                        if val > 120:
                            invalid_ages.append(d)

    return {"invalid_ids": list(set(invalid_ids)), "invalid_ages": list(set(invalid_ages))}
