"""
GPT-4o powered field extractor for ביטוח לאומי Form 283.

Takes raw OCR text from Azure Document Intelligence and maps it to
the exact JSON schema required by the assessment.
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import AzureOpenAI
from openai import APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

# ── JSON schema (mirrors the readme exactly) ──────────────────────────────────
EMPTY_SCHEMA: dict = {
    "lastName": "",
    "firstName": "",
    "idNumber": "",
    "gender": "",
    "dateOfBirth": {"day": "", "month": "", "year": ""},
    "address": {
        "street": "",
        "houseNumber": "",
        "entrance": "",
        "apartment": "",
        "city": "",
        "postalCode": "",
        "poBox": "",
    },
    "landlinePhone": "",
    "mobilePhone": "",
    "jobType": "",
    "dateOfInjury": {"day": "", "month": "", "year": ""},
    "timeOfInjury": "",
    "accidentLocation": "",
    "accidentAddress": "",
    "accidentDescription": "",
    "injuredBodyPart": "",
    "signature": "",
    "formFillingDate": {"day": "", "month": "", "year": ""},
    "formReceiptDateAtClinic": {"day": "", "month": "", "year": ""},
    "medicalInstitutionFields": {
        "healthFundMember": "",
        "natureOfAccident": "",
        "medicalDiagnoses": "",
    },
}

# ── Extraction prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert data extraction assistant for Israeli ביטוח לאומי (National Insurance Institute) Form 283 — a work-injury report form.

Extract all fields from the OCR text below and return a single JSON object. Follow every rule precisely.

--- GENERAL RULES ---
1. Return ONLY valid JSON. No markdown, no comments, no extra keys.
2. If a field is blank or not filled by the applicant, return "".
3. Never infer or hallucinate values. Extract only what is clearly present.
4. Dates → split into "day", "month", "year" string keys (e.g. day="02", month="11", year="1995").
5. gender → "זכר" or "נקבה" only. Return "" if unclear.

--- RULE: NAMES AND ID (top row of the form) ---
The form's very first data row contains THREE side-by-side fields:
  High x (right side of page): שם משפחה  → lastName
  Mid  x (centre of page):     שם פרטי   → firstName
  Low  x (left side of page):  מספר זהות → idNumber

These labels, and their handwritten values, all share roughly the same y-range
(within ±0.060 of each other). The OCR may emit all three handwritten values on
ONE line, or on closely adjacent lines, or even mixed into the label line.

Matching rule:
  1. Identify the y-band of the "שם משפחה" label (label line y).
  2. Collect ALL text tokens within ±0.060 y of that band.
  3. Separate printed labels (שם משפחה, שם פרטי, מספר זהות, ת.ז.) from user values.
  4. Among the remaining tokens, assign each to the label with the closest x:
     • Token with highest x  → lastName   (right side)
     • Token with middle x   → firstName  (centre)
     • Token with lowest x   → idNumber   (left side, must be a digit sequence — see ID rule)

CRITICAL: You MUST use x-coordinates to assign names — do NOT rely on linguistic
  intuition about which word "sounds like" a first name vs family name. Hebrew names
  can appear in either position; only the x-coordinate is authoritative.

CRITICAL: Do NOT stop after finding lastName.
  - If the same OCR line as lastName contains MORE Hebrew words, they belong to firstName.
  - A digit sequence on the same line belongs to idNumber.
  - firstName and idNumber MUST be extracted from the same y-band as lastName.

--- RULE: SPATIAL TOLERANCE ---
Every OCR line is prefixed with its normalised vertical position: [y:0.NNN]
and horizontal centre [x:0.NNN] (0.0 = left/top, 1.0 = right/bottom of page).
Lines with a y-difference ≤ 0.025 are on the SAME horizontal row.
Lines with a y-difference ≤ 0.060 are on ADJACENT rows.

Handwritten values are often placed slightly above or below their printed label.
When a printed label (e.g. "שם משפחה") appears to have no value on its own line:
  1. Find the label's y value.
  2. Scan all lines within ±0.060 of that y for candidate text.
  3. Any text at a nearby y that has no other label claiming it is the field value.
Apply this to ALL free-text fields (names, dates, address, jobType, description, etc.).

--- RULE: PHONE NUMBERS ---
All Israeli phone numbers start with "0" and are 9–10 digits long.
The blank form has a printed graphic artifact near the phone fields that OCR
sometimes reads as an extra digit before the leading "0" (often appearing as "6").

Correction — apply to landlinePhone and mobilePhone before storing:
  1. Strip spaces, dashes, and non-digit characters.
  2. The artifact may appear BEFORE or AFTER the leading "0" of the number:
     a. Artifact BEFORE "0" (string does not start with "0"):
        If removing the first digit gives a string starting with "0" with 9–10 digits
        → remove the first digit.
        Example: "6097656054" (10 chars) → "097656054" (9 digits ✓)
     b. Artifact AFTER "0" (string starts with "0" but has 11 digits):
        The second digit is the artifact. Remove it: result = digit[0] + digits[2:].
        If the result starts with "0" and has 10 digits → use it.
        Example: "08975423541" (11 chars) → "0" + "975423541" = "0975423541" (10 digits ✓)
  3. If the result already starts with "0" and has 9–10 digits → keep as-is.
  NOTE: mobilePhone must be exactly 10 digits; landlinePhone is 9–10 digits.

--- RULE: ID NUMBER ---
idNumber contains ONLY Arabic digits (0-9). It is NEVER Hebrew text or abbreviations.
The blank form contains printed Hebrew characters near the ID field (e.g. "ס״ב", "ת.ז.")
that are part of the form template — these are NOT the ID. Ignore any non-digit text.

Steps:
  1. Locate the label "מספר זהות" or "ת.ז." in the OCR.
  2. In the same y-band (±0.060), find the token that consists only of digits.
  3. Extract that digit sequence verbatim — do NOT discard it for a leading zero or
     non-standard length.
  4. If no digit sequence is found in the label's y-band, return "".

--- RULE: CHECKBOXES (most critical rule) ---
Azure OCR's geometric mapping ("=== Checkbox States ===") frequently fails on Israeli forms because filled checkboxes often float vertically, and Hebrew text labels are frequently merged into single text blocks. 

To guarantee accuracy, you MUST IGNORE the geometric labels in "=== Checkbox States ===" for the core checkbox fields. Instead, rely strictly on the topological sequence of the Unicode checkboxes (☐, ☑, v, x) in the raw text stream. Azure scans these specific forms in a precise column/row grid. 

Count the sequence of the boxes and group them into 3 clusters. Map the exact index of the selected box to the values below:

Cluster 1: Gender (2 boxes at the top)
- 1st box: "נקבה"
- 2nd box: "זכר"

Cluster 2: Accident Location (5 boxes in the middle)
- 1st box: "ת. דרכים בעבודה"
- 2nd box: "במפעל"
- 3rd box: "תאונה בדרך ללא רכב"
- 4th box: "ת. דרכים בדרך לעבודה/מהעבודה"
- 5th box: "אחר"

Cluster 3: Health Fund & Nature of Accident (7 boxes at the bottom)
- 1st box: "כללית" (Health)
- 2nd box: "תאונת עבודה" (Nature)
- 3rd box: "מאוחדת" (Health)
- 4th box: "מחלת מקצוע" (Nature)
- 5th box: "מכבי" (Health)
- 6th box: "אינו חבר בקופת חולים" (Health)
- 7th box: "לאומית" (Health)

HOW TO APPLY:
1. Find the checked mark (☑, v, x) in the raw text stream for the relevant section.
2. Count its position relative to the empty ☐ boxes in that exact section.
3. Match the numerical index to the canonical string mapped above.
(Example: If the text stream for the bottom section shows the 3rd box is checked, the answer is exactly "מאוחדת").

CANONICAL OPTIONS (match label text to closest option):
  gender            → "זכר" / "נקבה"
  accidentLocation  → "במפעל" / "ת. דרכים בעבודה" /
                       "ת. דרכים בדרך לעבודה/מהעבודה" / "תאונה בדרך ללא רכב" / "אחר"
  healthFundMember  → "כללית" / "מאוחדת" / "מכבי" / "לאומית"
  natureOfAccident  → "תאונת עבודה" / "מחלת מקצוע"

FORM LAYOUT — these groups occupy DIFFERENT physical sections of the form:
  gender           : top section (personal details, filled by claimant)
  accidentLocation : upper-middle section (section 3, filled by claimant)
  healthFundMember : bottom section (filled by clinic)
  natureOfAccident : bottom section (filled by clinic)
  ⚠️ NEVER mix options between groups. "במפעל", "בעבודה", "תאונה בדרך ללא רכב"
     belong ONLY to accidentLocation — they must NEVER appear in natureOfAccident
     or healthFundMember.

--- RULE: SIGNATURE ---
The signature field contains only what the claimant physically wrote in the חתימה box.
• If the claimant wrote their name or initials there (e.g. "רועי"), return that text.
• If the box is blank, return "".
• Do NOT copy the person's printed name from the top of the form into this field.
• Do NOT return "חתום" as a value.

--- FIELD MAPPING ---
שם משפחה          → lastName
שם פרטי           → firstName
מספר זהות / ת.ז.  → idNumber
מין               → gender
תאריך לידה        → dateOfBirth {day, month, year}
רחוב              → address.street
מספר בית          → address.houseNumber
כניסה             → address.entrance
דירה              → address.apartment
ישוב / עיר        → address.city
מיקוד             → address.postalCode
תא דואר           → address.poBox
טלפון קווי        → landlinePhone
טלפון נייד        → mobilePhone
סוג עבודה / מקצוע → jobType
תאריך הפגיעה      → dateOfInjury {day, month, year}
שעת הפגיעה        → timeOfInjury
מקום התאונה       → accidentLocation  [checkbox — see above]
כתובת מקום התאונה → accidentAddress
תיאור התאונה      → accidentDescription
האיבר שנפגע       → injuredBodyPart
חתימה             → signature
תאריך מילוי הטופס → formFillingDate {day, month, year}
תאריך קבלת הטופס  → formReceiptDateAtClinic {day, month, year}
חבר בקופת חולים   → medicalInstitutionFields.healthFundMember  [checkbox — see above]
מהות התאונה       → medicalInstitutionFields.natureOfAccident   [checkbox — see above]
אבחנות רפואיות    → medicalInstitutionFields.medicalDiagnoses

OUTPUT: one JSON object matching the schema. Nothing else.
"""

USER_PROMPT_TEMPLATE = """\
Below is the OCR text extracted from Form 283.
Extract all fields and return the JSON object.

--- OCR TEXT START ---
{ocr_text}
--- OCR TEXT END ---
"""


def _build_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def extract_fields(ocr_text: str, *, max_retries: int = 2) -> dict:
    """
    Call GPT-4o to extract form fields from OCR text.

    Returns a dict conforming to EMPTY_SCHEMA.
    Falls back to EMPTY_SCHEMA on unrecoverable errors.
    """
    client = _build_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT4O", "gpt-4o")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text)},
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            logger.info("GPT-4o extraction attempt %d/%d", attempt, max_retries + 1)
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0,          # deterministic extraction
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            logger.debug("Raw LLM response: %s", raw[:500])
            extracted = json.loads(raw)
            return _merge_with_schema(extracted)

        except (RateLimitError, APIConnectionError) as exc:
            logger.warning("Transient Azure OpenAI error (attempt %d): %s", attempt, exc)
            last_error = exc
            if attempt <= max_retries:
                import time
                time.sleep(2 ** attempt)  # exponential back-off
            continue

        except (APIError, json.JSONDecodeError) as exc:
            logger.error("Non-retryable error during extraction: %s", exc)
            last_error = exc
            break

    logger.error("All extraction attempts failed. Last error: %s", last_error)
    return dict(EMPTY_SCHEMA)


def _merge_with_schema(extracted: dict) -> dict:
    """
    Deep-merge the LLM output into EMPTY_SCHEMA so the result always has
    every key, even if the LLM omitted some.
    """
    import copy
    result = copy.deepcopy(EMPTY_SCHEMA)

    def _deep_merge(base: dict, override: dict) -> dict:
        for key, val in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                _deep_merge(base[key], val)
            elif key in base:
                # Normalise: ensure strings; None → ""
                base[key] = str(val).strip() if val is not None else ""
        return base

    merged = _deep_merge(result, extracted)
    _fix_phones(merged)
    return merged


def _fix_phone(raw: str) -> str:
    """
    Deterministically fix a phone number corrupted by the form's printed artifact.

    The artifact appears either BEFORE or AFTER the leading "0":
      - Before: "6097656054" (10 chars) → remove first digit → "097656054" (9 ✓)
      - After:  "08975423541" (11 chars) → remove second digit → "0975423541" (10 ✓)

    Valid Israeli phones: start with "0", 9–10 digits.
    """
    digits = re.sub(r"\D", "", raw)
    n = len(digits)

    # Already valid
    if digits.startswith("0") and n in (9, 10):
        return digits

    # Artifact BEFORE "0" — extra digit prepended: e.g. "6097656054" → "097656054"
    if not digits.startswith("0") and n in (10, 11):
        candidate = digits[1:]
        if candidate.startswith("0") and len(candidate) in (9, 10):
            return candidate

    # Artifact REPLACES "0" — first digit is wrong: e.g. "6554412742" → "0554412742"
    if not digits.startswith("0") and n in (9, 10):
        candidate = "0" + digits[1:]
        if len(candidate) in (9, 10):
            return candidate

    # Artifact AFTER "0" — extra digit inserted after leading 0: e.g. "08975423541" → "0975423541"
    if digits.startswith("0") and n == 11:
        candidate = digits[0] + digits[2:]
        if candidate.startswith("0") and len(candidate) == 10:
            return candidate

    return digits if digits else raw


def _fix_phones(data: dict) -> None:
    """Apply _fix_phone in-place to landlinePhone and mobilePhone."""
    for field in ("landlinePhone", "mobilePhone"):
        raw = data.get(field, "")
        if raw:
            data[field] = _fix_phone(raw)


def _to_date(d: dict) -> tuple[int, int, int] | None:
    """Convert a {day, month, year} dict to (year, month, day) int tuple, or None if incomplete."""
    try:
        y = int(d.get("year", "") or 0)
        m = int(d.get("month", "") or 0)
        day = int(d.get("day", "") or 0)
        if y and m and day:
            return (y, m, day)
    except (ValueError, TypeError):
        pass
    return None


def validate_extraction(data: dict) -> dict[str, list[str]]:
    """
    Validates extracted data.
    Returns {"warnings": [...], "errors": [...]}.
    Warnings = missing/suspect fields.
    Errors   = impossible date orderings or non-numeric date parts.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # ── Missing critical fields ────────────────────────────────────────────────
    for field in ("lastName", "firstName", "idNumber"):
        if not data.get(field):
            if field == "idNumber":
                warnings.append(
                    f"'{field}' is empty — may be unreadable handwriting or OCR failure. "
                    f"Ensure the ID number is clearly written and not mistaken for nearby labels."
                )
            else:
                if field == "lastName":
                    warnings.append(
                        f"'{field}' is empty — may be unreadable handwriting or OCR failure. "
                        f"Check the top right of the form for the family name, which should be near the 'שם משפחה' label."
                    )
                elif field == "firstName":
                     warnings.append(
                        f"'{field}' is empty — may be unreadable handwriting or OCR failure. "
                        f"Check the center of the top section for the first name, which should be near the 'שם פרטי' label."
                    )
    # ── ID number format ───────────────────────────────────────────────────────
    id_num = data.get("idNumber", "")
    if id_num:
        if not id_num.isdigit():
            warnings.append(
                f"ID number '{id_num}' contains non-digit characters — "
                f"Israeli ID must be exactly 9 digits."
            )
        elif len(id_num) != 9:
            warnings.append(
                f"ID number '{id_num}' has {len(id_num)} digit(s) — "
                f"Israeli ID must be exactly 9 digits."
            )

    # ── Numeric date parts ─────────────────────────────────────────────────────
    date_fields = ("dateOfBirth", "dateOfInjury", "formFillingDate", "formReceiptDateAtClinic")
    for date_field in date_fields:
        d = data.get(date_field, {})
        if isinstance(d, dict):
            for part in ("day", "month", "year"):
                val = d.get(part, "")
                if val and not re.match(r"^\d+$", val):
                    errors.append(f"{date_field}.{part} = '{val}' is not numeric.")

    # ── Date ordering logic ────────────────────────────────────────────────────
    dob    = _to_date(data.get("dateOfBirth", {}))
    doi    = _to_date(data.get("dateOfInjury", {}))
    ffd    = _to_date(data.get("formFillingDate", {}))
    frc    = _to_date(data.get("formReceiptDateAtClinic", {}))

    # dateOfBirth must be before injury / filling / receipt
    if dob and doi and dob >= doi:
        errors.append(
            f"Date of birth ({dob[2]:02d}/{dob[1]:02d}/{dob[0]}) is not before "
            f"Date of injury ({doi[2]:02d}/{doi[1]:02d}/{doi[0]})."
        )
    if dob and ffd and dob >= ffd:
        errors.append(
            f"Date of birth ({dob[2]:02d}/{dob[1]:02d}/{dob[0]}) is not before "
            f"Form filling date ({ffd[2]:02d}/{ffd[1]:02d}/{ffd[0]})."
        )
    if dob and frc and dob >= frc:
        errors.append(
            f"Date of birth ({dob[2]:02d}/{dob[1]:02d}/{dob[0]}) is not before "
            f"Form receipt date at clinic ({frc[2]:02d}/{frc[1]:02d}/{frc[0]})."
        )

    # dateOfInjury must be on or before formFillingDate
    if doi and ffd and doi > ffd:
        errors.append(
            f"Date of injury ({doi[2]:02d}/{doi[1]:02d}/{doi[0]}) is after "
            f"Form filling date ({ffd[2]:02d}/{ffd[1]:02d}/{ffd[0]}) — impossible."
        )

    # dateOfInjury must be on or before formReceiptDateAtClinic
    if doi and frc and doi > frc:
        errors.append(
            f"Date of injury ({doi[2]:02d}/{doi[1]:02d}/{doi[0]}) is after "
            f"Form receipt date at clinic ({frc[2]:02d}/{frc[1]:02d}/{frc[0]}) — impossible."
        )

    # formFillingDate must be on or before formReceiptDateAtClinic
    if ffd and frc and ffd > frc:
        errors.append(
            f"Form filling date ({ffd[2]:02d}/{ffd[1]:02d}/{ffd[0]}) is after "
            f"Form receipt date at clinic ({frc[2]:02d}/{frc[1]:02d}/{frc[0]}) — impossible."
        )

    return {"warnings": warnings, "errors": errors}