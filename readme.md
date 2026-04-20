<p align="center">
</p>

<h1 align="center">🤖 GenAI Developer Assessment — Solution</h1>

<p align="center">
  <strong>Azure Document Intelligence · GPT-4o · ADA-002 · FastAPI · Streamlit · Gradio</strong>
</p>

---

## Overview

This solution is built in two self-contained phases, each solving a real-world GenAI problem common in enterprise document-processing and customer-service domains.

| Phase | Problem | Stack |
|-------|---------|-------|
| **1** | Extract structured data from filled insurance forms | Azure DI + GPT-4o + Streamlit |
| **2** | Conversational chatbot that answers HMO service questions for a specific user profile | FastAPI + GPT-4o + ADA-002 + Gradio |

---

## Phase 1 — Form 283 Field Extraction

### Motivation

ביטוח לאומי (National Insurance Institute) Form 283 is a hand-filled work-injury report. Digitising it manually is slow and error-prone. The goal is a fully automated pipeline: scan → extract → validate → structured JSON, ready for downstream systems.

### Architecture

```
User uploads PDF/image
        │
        ▼
┌──────────────────────────────┐
│  Azure Document Intelligence │  prebuilt-layout model
│  (ocr_processor.py)          │  → lines with [y x] coordinates
│                              │  → selection-mark states + spans
│                              │  → table data
└──────────────┬───────────────┘
               │  Rich structured text
               ▼
┌──────────────────────────────┐
│  GPT-4o Extractor            │  System prompt with spatial rules
│  (llm_extractor.py)          │  → 24-field JSON schema
│                              │  → phone correction
│                              │  → checkbox resolution
└──────────────┬───────────────┘
               │  Raw JSON
               ▼
┌──────────────────────────────┐
│  Validation                  │  Date ordering, ID format,
│  (validate_extraction)       │  phone length, completeness
└──────────────┬───────────────┘
               │
               ▼
        Streamlit UI
   (raw JSON + field view + download)
```

### Screenshot

![Phase 1 UI](/files/phase1_screenshot.png)

The UI shows:
- **Left panel** — uploaded document preview
- **Right panel** — extracted fields in two views: raw JSON and a human-readable field-by-field breakdown
- **Validation report** — errors (e.g. date order violations) and warnings (e.g. missing ID) surfaced inline
- **Download button** — exports the final JSON

### Key Engineering Challenges

#### 1. Spatial Layout of a Hebrew RTL Form

Standard OCR returns text in reading order, losing the 2-D layout that tells you *which* handwritten word belongs to *which* printed label. A single line in the top row of Form 283 can contain three different fields (last name, first name, ID) side by side.

**Solution:** Every OCR line is prefixed with its normalised page coordinates `[y:0.NNN x:0.NNN]`. The LLM prompt teaches GPT-4o to reason about horizontal (`x`) distance to assign tokens to the correct label — highest x → `lastName`, middle x → `firstName`, lowest x → `idNumber` — regardless of reading order.

#### 2. Checkbox Extraction

Form 283 has four checkbox groups (gender, accident location, health fund, nature of accident).

**Solution (two-layer approach):**

*Primary — inter-mark text segments:*
Azure DI embeds `:selected:` / `:unselected:` tokens at exact byte offsets in `result.content`. For each mark, the code extracts:
- `before` = text between the **previous** mark token and this one (≤5 words)
- `after`  = text between this mark and the **next** one (≤5 words)

The option label sits in exactly one of these two slots. By looking at the pattern across all marks in a group (consistent `before` or consistent `after` convention), GPT-4o resolves the label unambiguously.

*Fallback — spatial word matching:*
If span offsets are unavailable, the nearest Hebrew word within ±0.20 inches vertically and ±1.80 inches horizontally is taken as the label (RTL-aware).

#### 3. Phone Number OCR Artifacts

The blank form contains a printed graphic near the phone fields. OCR sometimes reads this as a stray digit prepended or inserted into the phone number.

**Solution:** `_fix_phone()` applies two deterministic corrections before storing:
- Artifact **before** leading zero (e.g. `"6097656054"`) → strip first digit
- Artifact **after** leading zero (e.g. `"08975423541"`, 11 digits) → `"0" + digits[2:]`

Both corrections are guarded by length and leading-zero checks so valid numbers are never mutated.

---

## Phase 2 — HMO Medical Services Chatbot

### Motivation

Patients frequently need to know what medical services they are entitled to under their specific health fund and membership tier. The information exists in HTML tables but is spread across multiple files and tiers. A chatbot that collects user context first, then retrieves only the relevant slice, provides a fast and personalised experience.

### Architecture

```
Gradio Frontend (stateless client)
   │  sends: full message history + user profile + phase
   ▼
FastAPI Backend  /chat
   │
   ├── Phase: collection
   │     └── GPT-4o with function-calling tool (submit_user_profile)
   │           Collects name, ID, gender, age, HMO, card number, tier
   │           Validates: 9-digit IDs, age 0–120
   │           Asks user to confirm before submitting
   │
   └── Phase: qa
         └── Retrieve from KnowledgeBase (ADA-002 cosine similarity)
               Filter: user's HMO + tier only
               GPT-4o answers based solely on retrieved context
               Appends contact info chunk for the relevant category
```

All conversation history and user profile live on the **client side** (Gradio `gr.State`). The backend is fully stateless — it scales horizontally with no shared memory.

### Screenshot

![Phase 2 UI](/files/phase2_screenshot.png)

The UI shows:
- **Left panel** — chat window with the bot guiding the user through profile collection, then switching to Q&A
- **Right panel** — live user profile card showing collected fields and the current phase badge (Collection / Q&A)
- Language auto-detection: the bot responds in Hebrew or English to match the user

### Key Engineering Challenges

#### 1. Stateless Multi-User Architecture

Maintaining server-side sessions limits scalability and adds operational complexity.

**Solution:** Every `/chat` request carries the full message history and profile. The backend is a pure function: `(messages, profile, phase) → (reply, updated_profile, new_phase)`. Any number of replicas can serve requests with no coordination.

#### 2. LLM-Driven Profile Collection (no hardcoded Q&A)

The assignment explicitly forbids hardcoded question flows. The system must feel natural and handle corrections mid-conversation.

**Solution:** GPT-4o is given a `submit_user_profile` function-calling tool. It decides when to call the tool (only when all fields are filled and the user has confirmed). Until then, it asks conversationally, corrects misunderstandings, and validates inputs inline. Field validators catch invalid IDs or out-of-range ages and let the LLM report them naturally.

#### 3. RAG with HMO/Tier Filtering

The knowledge base contains data for 3 HMOs × 3 tiers. Returning chunks from other HMOs or tiers would mislead the user.

**Solution:** Each `KnowledgeChunk` is tagged with `(hmo, tier, category)`. Retrieval applies a hard filter before cosine similarity ranking. Only if zero chunks pass the filter does it fall back to unfiltered results (preventing silent failures).

#### 4. Multi-Language Consistency

Users switching between Hebrew and English mid-conversation can confuse the model into mixing languages.

**Solution:** `_last_user_language()` detects the script of the most recent user turn. A `_lang_hint()` string is injected into every prompt: *"Respond in Hebrew only"* or *"Respond in English only"*, ensuring the bot never mixes languages within a single reply.

---

## Testing Strategy

Tests live in `tests/` and are split into two sections per phase.

### Design Principle: No Hardcoded Values

All test inputs are **data-driven**: loaded from the ground-truth example files (`ex1_correct.json`, `ex2_correct.json`, `ex3_correct.json`) and the example conversation transcripts (`phase2_ex*.txt`). This means the test suite remains valid if the example files are replaced or extended — no test knows about a specific name, ID, or phone number.

### Phase 1 Unit Tests (`test_phase1.py`)

Run without any Azure credentials:

```bash
pytest tests/test_phase1.py -v -m "not integration"
```

| Class | What it tests |
|-------|--------------|
| `TestFixPhone` | `_fix_phone()` leaves real phones unchanged; repairs OCR artifacts (prepend, insert) on real phone numbers loaded from fixtures |
| `TestValidateExtraction` | `validate_extraction()` on a structurally-valid record from `ex2_correct.json`; each test modifies exactly one field to trigger a specific warning or error |

**Key design decisions:**
- `valid_extraction_base` fixture builds a clean record from real data (`ex2_correct.json`) with the receipt date pushed one year forward so all chronological constraints are satisfied.
- Phone corruption tests simulate the two known artifact patterns on actual phone numbers from the JSON files — the corrupted values are never invented.
- Date violation tests compute violation years relative to the real birth year (`birth_year - 5`) rather than hardcoding a year.

### Phase 1 Integration / Accuracy Tests

Run the full OCR → GPT-4o pipeline against the three sample PDFs:

```bash
pytest tests/test_phase1.py -v -s -m integration --run-integration
```

Produces a detailed accuracy report:

```
======================================================================
  PHASE 1 — END-TO-END ACCURACY REPORT
======================================================================
  Overall : 47/51 fields correct  (92.2%)

  Example 1 : 16/17 (94.1%)
  Example 2 : 17/17 (100.0%)
  Example 3 : 14/17 (82.4%)

  Error breakdown by category:
    CHECKBOX       :  1 field(s)
    DATE           :  2 field(s)
======================================================================
```

Error categories: `CHECKBOX`, `PHONE`, `NAME_SWAP`, `ID`, `FIELD_EMPTY`, `WRONG_VALUE`, `DATE`.

### Phase 2 Unit Tests (`test_phase2.py`)

Run without Azure credentials:

```bash
pytest tests/test_phase2.py -v -m "not integration"
```

| Class | What it tests |
|-------|--------------|
| `TestLastUserLanguage` | Language detection on real messages from example conversations |
| `TestLangHint` | Prompt hints generated for Hebrew/English turns |
| `TestTransitionMessage` | Collection→QA transition message uses real profile fields from fixtures |
| `TestBuildValidatedProfile` | Profile validation with real invalid IDs/ages extracted from rejected-input turns in example conversations |
| `TestIsComplete` | `is_complete()` on profiles parsed from fixture transcripts |

### Phase 2 Replay Tests (Integration)

Replay all five example conversations against the live backend, comparing actual bot responses to the reference responses in the `.txt` files using keyword similarity scoring.

```bash
pytest tests/test_phase2.py -v -s -m integration --run-integration
```

Output shows per-turn comparison:

```
[PASS] turn 2  score=91%
       ref   : 'שלום! אני כאן לעזור לך...'
       actual: 'שלום! בשמחה אסייע לך...'
[FAIL] turn 4  score=61%
       ref   : 'מספר הזהות שלך אינו תקין'
       actual: 'אנא הכנס מספר זהות בן 9 ספרות'
```

A per-example pass threshold of **80%** is applied: if ≥80% of turns in an example pass the keyword similarity check, the entire example test passes. This tolerates natural LLM paraphrase variation while catching systematic regressions.

---

## Project Structure

```
Home-Assignment-GenAI-KPMG/
│
├── phase1/
│   ├── app.py               # Streamlit UI
│   ├── ocr_processor.py     # Azure DI wrapper — rich spatial OCR output
│   └── llm_extractor.py     # GPT-4o extraction, phone fix, validation
│
├── phase2/
│   ├── frontend.py          # Gradio UI (client-side state management)
│   └── backend/
│       ├── main.py          # FastAPI stateless microservice
│       ├── knowledge_base.py# HTML parsing + ADA-002 embeddings + retrieval
│       └── models.py        # Pydantic models (UserProfile, ChatRequest/Response)
│
├── tests/
│   ├── conftest.py          # Shared fixtures (loads real data from files/)
│   ├── test_phase1.py       # Phase 1 unit + integration tests
│   └── test_phase2.py       # Phase 2 unit + replay tests
│
├── phase1_data/             # Sample Form 283 PDFs
├── phase2_data/             # HMO service HTML knowledge base
├── files/                   # Ground-truth JSONs, example conversations, screenshots
├── requirements.txt
├── pytest.ini
└── SETUP.md                 # Environment setup and run instructions
```

---

## Setup & Running

See [SETUP.md](SETUP.md) for the full instructions. Quick start:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Azure credentials
cp .env.example .env   # fill in your credentials

# 3. Run Phase 1
streamlit run phase1/app.py

# 4. Run Phase 2 (two terminals)
python -m uvicorn phase2.backend.main:app --host 0.0.0.0 --port 8000 --reload
python phase2/frontend.py
```

---

## Next Steps & Known Limitations

### 1. Checkbox Extraction — Further Improvement

The current approach (inter-mark text segments) is a significant improvement over raw context windows, but it is still dependent on Azure DI's reading-order algorithm placing the label in a predictable slot relative to the mark. On unusual form layouts, this can still fail.

**Ideas:**
- Train a custom Azure DI model on Form 283 specifically (few-shot with labeled checkboxes)
- Use Azure DI's `boundingRegions` to correlate each mark's bounding box with the nearest text bounding box geometrically, independent of reading order
- Post-process: if GPT-4o returns an `accidentLocation` value that is a known `healthFundMember` option (or vice versa), flag it as a group-mix error and retry with a targeted correction prompt

### 2. Manual JSON Correction UI

Currently, the extracted JSON is read-only — users can view and download it but not correct mistakes inline.

**Proposed feature:** After extraction, render each field as an editable text input pre-filled with the extracted value. A "Re-validate" button re-runs `validate_extraction()` on the edited values. The corrected JSON is what gets downloaded. This is especially valuable for checkbox fields and dates where OCR errors are most common.

### 3. Test Coverage Expansion

- Add **property-based tests** (e.g. with Hypothesis) for `_fix_phone()` — generate random digit sequences and assert invariants (idempotency, length preservation for valid numbers)
- Add **more example PDFs** to the accuracy tests — currently only three examples. Edge cases like partially-filled forms, forms with corrections/whiteout, or English-filled forms are not covered
- Add **contract tests** for the FastAPI `/chat` endpoint (request/response schema, phase transition rules, statelessness guarantee)
- Add a **performance regression test** for the knowledge base: embedding load time and retrieval latency should stay below a threshold as the HTML data grows

### 4. Phase 2 — Knowledge Base Enhancements

- Add **hybrid search**: combine BM25 keyword matching with ADA-002 semantic similarity for better recall on exact HMO/tier names
- Implement **answer attribution**: include the source chunk (category + file) in the response so users know where the information comes from
- Add **out-of-scope detection**: if the user asks something not covered by the knowledge base, the bot should say so rather than hallucinating or returning a low-confidence answer

### 5. Observability

- Structured logging (JSON lines) for every extraction attempt and chat turn, suitable for ingestion into Azure Monitor or Elasticsearch
- Accuracy dashboard: track per-field error rates over time to detect drift if form templates change
- Latency monitoring for OCR, embedding, and LLM calls separately

---
