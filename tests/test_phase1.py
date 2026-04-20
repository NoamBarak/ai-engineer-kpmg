"""
Phase 1 Tests
=============

SECTION A — Unit tests (no Azure needed)
  Tests pure Python functions: _fix_phone, validate_extraction.

  All test inputs come from the example files (ex*_correct.json, ex*_wrong.json)
  rather than invented values, so the tests remain valid if the example files
  are replaced in the future.

  Run at any time:
      pytest tests/test_phase1.py -v -m "not integration"

SECTION B — End-to-end accuracy tests (require Azure credentials)
  Runs each 283_exN.pdf through the full OCR → GPT-4o pipeline,
  compares every extracted field against exN_correct.json, and
  prints a detailed accuracy report.
  Run with:
      pytest tests/test_phase1.py -v -s -m integration --run-integration

Error categories reported:
  CHECKBOX    — gender / accidentLocation / healthFundMember / natureOfAccident
  PHONE       — phone numbers corrupted by OCR artifact
  NAME_SWAP   — firstName / lastName transposed
  ID          — idNumber wrong or empty
  FIELD_EMPTY — field should have a value but extracted as ""
  WRONG_VALUE — non-empty but incorrect
  DATE        — any sub-field of a date object is wrong
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from phase1.llm_extractor import _fix_phone, validate_extraction


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — Unit tests (no Azure)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Helper used across TestFixPhone ──────────────────────────────────────────

def _first_phone(correct_jsons: dict) -> str:
    """Return the first non-empty phone number found in the correct JSONs."""
    for _ex_id, data in sorted(correct_jsons.items()):
        for field in ("mobilePhone", "landlinePhone"):
            phone = data.get(field, "")
            if phone:
                return phone
    pytest.skip("No phone numbers found in correct JSONs")


class TestFixPhone:
    """
    Tests _fix_phone() using phone numbers loaded from ex*_correct.json and
    ex*_wrong.json.  No phone numbers are hard-coded; if the example files are
    replaced the tests automatically use the new values.
    """

    # ── Data-driven: real phones must survive unchanged ───────────────────────

    def test_all_correct_phones_returned_unchanged(self, correct_jsons):
        """Every non-empty phone in ex*_correct.json must pass through unmodified."""
        tested = 0
        for ex_id, data in sorted(correct_jsons.items()):
            for field in ("landlinePhone", "mobilePhone"):
                phone = data.get(field, "")
                if phone:
                    assert _fix_phone(phone) == phone, (
                        f"ex{ex_id} {field}: valid number {phone!r} was mutated"
                    )
                    tested += 1
        assert tested > 0, "No phone numbers found in correct JSONs — check fixture"

    # ── Data-driven: OCR-corrupted phones must be repaired ────────────────────

    def test_corrupted_phones_fixed_to_correct(self, correct_jsons, wrong_jsons):
        """
        For every phone that differs between ex*_wrong.json and ex*_correct.json,
        _fix_phone() applied to the wrong value must produce the correct value.
        """
        found_corruption = False
        for ex_id in sorted(correct_jsons.keys()):
            if ex_id not in wrong_jsons:
                continue
            for field in ("landlinePhone", "mobilePhone"):
                correct_val = correct_jsons[ex_id].get(field, "")
                wrong_val   = wrong_jsons[ex_id].get(field, "")
                if wrong_val and correct_val and wrong_val != correct_val:
                    result = _fix_phone(wrong_val)
                    assert result == correct_val, (
                        f"ex{ex_id} {field}: _fix_phone({wrong_val!r}) → {result!r}, "
                        f"expected {correct_val!r}"
                    )
                    found_corruption = True
        assert found_corruption, (
            "No corrupted phone numbers found in wrong JSONs. "
            "Ensure ex*_wrong.json files are present and contain phone artifacts."
        )

    # ── Data-driven: simulated artifact patterns on real phone numbers ────────

    def test_artifact_digit_prepended_before_leading_zero(self, correct_jsons):
        """
        Simulates the 'prepend' OCR artifact: a stray digit is added before the
        leading '0'.  _fix_phone() must strip it and recover the original number.
        """
        for ex_id, data in sorted(correct_jsons.items()):
            for field in ("mobilePhone", "landlinePhone"):
                phone = data.get(field, "")
                if phone and phone.startswith("0"):
                    corrupted = "6" + phone      # e.g. "6097656054"
                    assert _fix_phone(corrupted) == phone, (
                        f"ex{ex_id} {field}: prepend artifact on {phone!r} not recovered"
                    )

    def test_artifact_digit_inserted_after_leading_zero(self, correct_jsons):
        """
        Simulates the 'insert' OCR artifact: a stray digit is inserted right
        after the leading '0'.  _fix_phone() must remove it.
        """
        for ex_id, data in sorted(correct_jsons.items()):
            for field in ("mobilePhone", "landlinePhone"):
                phone = data.get(field, "")
                if phone and phone.startswith("0"):
                    corrupted = "0" + "8" + phone[1:]   # e.g. "08502474947"
                    assert _fix_phone(corrupted) == phone, (
                        f"ex{ex_id} {field}: insert artifact on {phone!r} not recovered"
                    )

    # ── Structural edge-case tests (algorithm behaviour, not user data) ───────

    def test_empty_string_returned_unchanged(self):
        assert _fix_phone("") == ""

    def test_spaces_in_valid_number_stripped(self, correct_jsons):
        """A real phone number with spaces inserted must still be normalised."""
        phone = _first_phone(correct_jsons)
        spaced = " ".join(list(phone))          # "0 5 5 4 4 1 2 7 4 2"
        assert _fix_phone(spaced) == phone

    def test_dashes_in_valid_number_stripped(self, correct_jsons):
        """A real phone number with a dash in the middle must be normalised."""
        phone = _first_phone(correct_jsons)
        mid    = len(phone) // 2
        dashed = phone[:mid] + "-" + phone[mid:]
        assert _fix_phone(dashed) == phone


class TestValidateExtraction:
    """
    Tests validate_extraction() using records derived from ex*_correct.json.
    Personal data (names, ID, address, phone) is never invented — it is loaded
    from the fixture.  Only the single field under test is modified.
    """

    def test_clean_record_no_issues(self, valid_extraction_base):
        result = validate_extraction(valid_extraction_base)
        assert result["errors"] == [] and result["warnings"] == [], (
            f"Expected clean record but got: errors={result['errors']}, "
            f"warnings={result['warnings']}"
        )

    def test_missing_last_name_warns(self, valid_extraction_base):
        d = copy.deepcopy(valid_extraction_base)
        d["lastName"] = ""
        assert any("lastName" in w for w in validate_extraction(d)["warnings"])

    def test_missing_first_name_warns(self, valid_extraction_base):
        d = copy.deepcopy(valid_extraction_base)
        d["firstName"] = ""
        assert any("firstName" in w for w in validate_extraction(d)["warnings"])

    def test_missing_id_warns(self, valid_extraction_base):
        d = copy.deepcopy(valid_extraction_base)
        d["idNumber"] = ""
        warns = validate_extraction(d)["warnings"]
        assert any("idNumber" in w for w in warns)

    def test_non_digit_id_warns(self, valid_extraction_base):
        d = copy.deepcopy(valid_extraction_base)
        d["idNumber"] = "סב"
        warns = validate_extraction(d)["warnings"]
        assert any("non-digit" in w.lower() for w in warns)

    def test_extra_digit_in_id_warns(self, valid_extraction_base):
        """Adding a digit to the real 9-digit ID makes it 10 digits → warning."""
        d = copy.deepcopy(valid_extraction_base)
        d["idNumber"] = d["idNumber"] + "0"     # real 9-digit ID → 10 digits
        warns = validate_extraction(d)["warnings"]
        assert any("9 digit" in w.lower() for w in warns)

    def test_injury_before_birth_errors(self, valid_extraction_base):
        d = copy.deepcopy(valid_extraction_base)
        birth_year = int(d["dateOfBirth"]["year"])
        d["dateOfInjury"] = {"day": "01", "month": "01", "year": str(birth_year - 5)}
        assert len(validate_extraction(d)["errors"]) > 0

    def test_filling_before_birth_errors(self, valid_extraction_base):
        d = copy.deepcopy(valid_extraction_base)
        birth_year = int(d["dateOfBirth"]["year"])
        d["formFillingDate"] = {"day": "01", "month": "01", "year": str(birth_year - 5)}
        assert len(validate_extraction(d)["errors"]) > 0

    def test_receipt_before_injury_errors(self, valid_extraction_base):
        d = copy.deepcopy(valid_extraction_base)
        injury_year = int(d["dateOfInjury"]["year"])
        d["formReceiptDateAtClinic"] = {
            "day": "01", "month": "01", "year": str(injury_year - 1),
        }
        assert len(validate_extraction(d)["errors"]) > 0

    def test_non_numeric_date_part_errors(self, valid_extraction_base):
        d = copy.deepcopy(valid_extraction_base)
        d["dateOfBirth"]["month"] = "אוגוסט"
        assert any("not numeric" in e.lower() for e in validate_extraction(d)["errors"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — End-to-end accuracy tests (require Azure)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Error categorisation helpers ──────────────────────────────────────────────

class ErrorCategory(str, Enum):
    CHECKBOX    = "CHECKBOX"
    PHONE       = "PHONE"
    NAME_SWAP   = "NAME_SWAP"
    ID          = "ID"
    FIELD_EMPTY = "FIELD_EMPTY"
    WRONG_VALUE = "WRONG_VALUE"
    DATE        = "DATE"


_CHECKBOX_FIELDS = {
    "gender", "accidentLocation",
    "medicalInstitutionFields.healthFundMember",
    "medicalInstitutionFields.natureOfAccident",
}
_PHONE_FIELDS  = {"landlinePhone", "mobilePhone"}
_DATE_PREFIXES = {"dateOfBirth", "dateOfInjury", "formFillingDate", "formReceiptDateAtClinic"}


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict to dot-notation keys with string values."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = str(v).strip()
    return out


def _categorize(
    field: str, actual: str,
    flat_correct: dict[str, str],
) -> ErrorCategory:
    if field in _CHECKBOX_FIELDS:
        return ErrorCategory.CHECKBOX
    if field in _PHONE_FIELDS:
        return ErrorCategory.PHONE
    if field == "idNumber":
        return ErrorCategory.ID
    if field == "firstName" and actual == flat_correct.get("lastName", ""):
        return ErrorCategory.NAME_SWAP
    if field == "lastName" and actual == flat_correct.get("firstName", ""):
        return ErrorCategory.NAME_SWAP
    for dp in _DATE_PREFIXES:
        if field.startswith(dp):
            return ErrorCategory.DATE
    if actual == "":
        return ErrorCategory.FIELD_EMPTY
    return ErrorCategory.WRONG_VALUE


# ── Accumulate results across the three parametrised tests ───────────────────
_e2e_results: list[dict] = []


@pytest.fixture(autouse=True, scope="module")
def _print_e2e_summary():
    """Print accuracy report after all e2e tests finish."""
    yield
    if not _e2e_results:
        return

    total   = len(_e2e_results)
    correct = sum(1 for r in _e2e_results if r["match"])
    pct     = 100.0 * correct / total if total else 0

    by_ex:  dict[str, dict]  = {}
    by_cat: dict[str, list]  = defaultdict(list)

    for r in _e2e_results:
        ex = r["example"]
        by_ex.setdefault(ex, {"total": 0, "correct": 0, "errors": []})
        by_ex[ex]["total"] += 1
        if r["match"]:
            by_ex[ex]["correct"] += 1
        else:
            by_ex[ex]["errors"].append(r)
            if r["category"]:
                by_cat[r["category"]].append(r)

    print("\n")
    print("=" * 72)
    print("  PHASE 1 — END-TO-END ACCURACY REPORT")
    print("=" * 72)
    print(f"  Overall : {correct}/{total} fields correct  ({pct:.1f}%)")
    print()

    for ex in sorted(by_ex.keys()):
        info = by_ex[ex]
        ex_pct = 100.0 * info["correct"] / info["total"] if info["total"] else 0
        print(f"  Example {ex} : {info['correct']}/{info['total']} ({ex_pct:.1f}%)")

    print()
    print("  Error breakdown by category:")
    any_errors = False
    for cat in ErrorCategory:
        errors = by_cat.get(cat.value, [])
        if errors:
            any_errors = True
            print(f"    {cat.value:14s} : {len(errors):2d} field(s)")
            for e in errors:
                print(f"        ex{e['example']}  {e['field']:50s}"
                      f"  expected={e['expected']!r:25s}  got={e['actual']!r}")

    if not any_errors:
        print("    ✅  No errors — all fields match ground truth!")
    print("=" * 72)


@pytest.mark.integration
@pytest.mark.parametrize("example_id", ["1", "2", "3"])
def test_phase1_e2e_accuracy(example_id: str, correct_jsons, pdf_paths, run_integration):
    """
    Run 283_ex{N}.pdf through the full OCR → GPT-4o pipeline,
    then compare every field against ex{N}_correct.json.

    Requires --run-integration and valid Azure env vars.
    """
    if not run_integration:
        pytest.skip("Pass --run-integration to run this test (requires Azure credentials).")

    from phase1.ocr_processor import extract_text_from_bytes
    from phase1.llm_extractor import extract_fields

    pdf_path = pdf_paths[example_id]
    if not pdf_path.exists():
        pytest.skip(f"PDF not found: {pdf_path}")

    # ── Run the full pipeline ─────────────────────────────────────────────────
    with open(pdf_path, "rb") as fh:
        raw_bytes = fh.read()

    ocr_text  = extract_text_from_bytes(raw_bytes, "application/pdf")
    extracted = extract_fields(ocr_text)

    correct  = _flatten(correct_jsons[example_id])
    actual   = _flatten(extracted)

    mismatches: list[str] = []
    total = len(correct)

    for field, expected_val in correct.items():
        actual_val = actual.get(field, "")
        match      = (actual_val == expected_val)
        category   = _categorize(field, actual_val, correct) if not match else None

        _e2e_results.append({
            "example":  example_id,
            "field":    field,
            "expected": expected_val,
            "actual":   actual_val,
            "match":    match,
            "category": category.value if category else None,
        })

        if not match:
            mismatches.append(
                f"\n  [{category.value if category else '?'}] "
                f"field={field!r}  "
                f"expected={expected_val!r}  got={actual_val!r}"
            )

    passed_count = total - len(mismatches)
    pct = 100.0 * passed_count / total if total else 0

    assert mismatches == [], (
        f"Example {example_id}: {passed_count}/{total} fields correct ({pct:.1f}%). "
        f"Failed fields:{' '.join(mismatches)}"
    )
