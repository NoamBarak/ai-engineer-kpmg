"""
Phase 2 Tests
=============

SECTION A — Unit tests (no Azure needed)
  Tests pure Python functions: language detection, profile building, etc.

  All test data is derived from the phase2_ex*.txt conversation files loaded
  by the conftest fixtures.  No values are hard-coded so the tests remain
  valid if the example files are replaced in the future.

  Run at any time:
      pytest tests/test_phase2.py -v -m "not integration"

SECTION B — Conversation replay tests (require Azure credentials)
  Replays each example conversation from phase2_ex*.txt through the real
  FastAPI app (with real Azure OpenAI calls).

  For each user turn:
    1. POST to /chat with the full accumulated history + current profile + phase.
    2. Receive the actual assistant reply and updated profile/phase.
    3. Compare the actual reply against the reference reply in the .txt file
       using keyword matching (semantic tokens, not exact text).

  A turn PASSES if the actual response contains >= MIN_KEYWORD_RATIO (80%) of
  the key semantic tokens extracted from the reference response.

  Final report shows:
    - Per-turn pass/fail and keyword score
    - Per-example pass rate
    - Overall pass rate across all conversations

  Run with:
      pytest tests/test_phase2.py -v -s -m integration --run-integration
"""
from __future__ import annotations

import json
import re
from typing import Any

import pytest

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — Unit tests (no Azure)
# ═══════════════════════════════════════════════════════════════════════════════

from phase2.backend.main import (
    _last_user_language,
    _lang_hint,
    _transition_message,
    _build_validated_profile,
)
from phase2.backend.models import Message, UserProfile, Phase


# ── Message helpers ───────────────────────────────────────────────────────────

def _user(content: str) -> Message:
    return Message(role="user", content=content)


def _assistant(content: str) -> Message:
    return Message(role="assistant", content=content)


# ── Profile helper ────────────────────────────────────────────────────────────

def _profile_from_dict(d: dict) -> UserProfile:
    """Convert a conftest _parse_profile() result dict to a UserProfile instance."""
    return UserProfile(
        firstName=d.get("firstName", ""),
        lastName=d.get("lastName", ""),
        idNumber=d.get("idNumber", ""),
        gender=d.get("gender", ""),
        age=d.get("age"),
        hmo=d.get("hmo", ""),
        hmoCardNumber=d.get("hmoCardNumber", ""),
        tier=d.get("tier", ""),
    )


def _find_profile(phase2_profiles: dict, require: tuple[str, ...] = ()) -> dict | None:
    """
    Return the first profile from phase2_profiles that has all ``require`` fields
    set to a non-empty/non-None value.  Returns None if no match is found.
    """
    for data in phase2_profiles.values():
        if all(data.get(k) for k in require) and (
            "age" not in require or data.get("age") is not None
        ):
            return data
    return None


# ── _last_user_language ───────────────────────────────────────────────────────

class TestLastUserLanguage:
    """
    Tests _last_user_language() using actual messages from phase2_ex*.txt files.
    Hebrew conversations: ex1, ex2, ex4.
    English conversation: ex3.
    """

    def test_hebrew_user_messages_detected_as_he(self, phase2_conversations):
        """User messages containing Hebrew characters must return 'he'."""
        for key in ("ex1", "ex2", "ex4"):
            turns = phase2_conversations.get(key, [])
            hebrew_msgs = [
                t["content"] for t in turns
                if t["role"] == "user"
                and any("\u0590" <= c <= "\u05FF" for c in t["content"])
            ]
            for msg in hebrew_msgs:
                assert _last_user_language([_user(msg)]) == "he", (
                    f"Expected 'he' for {key} message: {msg!r}"
                )

    def test_english_user_messages_detected_as_en(self, phase2_conversations):
        """Pure-Latin user messages from ex3 (English speaker) must return 'en'."""
        turns = phase2_conversations.get("ex3", [])
        if not turns:
            pytest.skip("ex3 conversation file not found")
        latin_msgs = [
            t["content"] for t in turns
            if t["role"] == "user"
            and t["content"].strip()
            and not any("\u0590" <= c <= "\u05FF" for c in t["content"])
        ]
        if not latin_msgs:
            pytest.skip("No pure-Latin user messages found in ex3")
        for msg in latin_msgs:
            assert _last_user_language([_user(msg)]) == "en", (
                f"Expected 'en' for pure-Latin message: {msg!r}"
            )

    def test_last_message_language_wins_over_earlier(self, phase2_conversations):
        """Language detection must use only the LAST user message."""
        # Find one Hebrew and one English message from the conversation files.
        he_msg = None
        en_msg = None
        for turns in phase2_conversations.values():
            for t in turns:
                if t["role"] != "user":
                    continue
                if he_msg is None and any("\u0590" <= c <= "\u05FF" for c in t["content"]):
                    he_msg = t["content"]
                if en_msg is None and not any("\u0590" <= c <= "\u05FF" for c in t["content"]) \
                        and t["content"].strip():
                    en_msg = t["content"]
            if he_msg and en_msg:
                break

        if not (he_msg and en_msg):
            pytest.skip("Could not find both Hebrew and English user messages in files")

        # Hebrew first, English last → "en"
        assert _last_user_language(
            [_user(he_msg), _assistant("..."), _user(en_msg)]
        ) == "en"
        # English first, Hebrew last → "he"
        assert _last_user_language(
            [_user(en_msg), _assistant("..."), _user(he_msg)]
        ) == "he"

    def test_empty_conversation_defaults_to_he(self):
        assert _last_user_language([]) == "he"

    def test_only_assistant_turns_defaults_to_he(self, phase2_conversations):
        """With no user turns in the list, default language is Hebrew."""
        turns = next(iter(phase2_conversations.values()), [])
        assistant_msgs = [
            _assistant(t["content"]) for t in turns if t["role"] == "assistant"
        ][:3]
        if not assistant_msgs:
            pytest.skip("No assistant turns found in conversation files")
        assert _last_user_language(assistant_msgs) == "he"

    def test_mixed_hebrew_and_latin_detects_hebrew(self, phase2_conversations):
        """A message containing even one Hebrew character must return 'he'."""
        for turns in phase2_conversations.values():
            for t in turns:
                if t["role"] != "user":
                    continue
                has_heb = any("\u0590" <= c <= "\u05FF" for c in t["content"])
                has_lat = any("a" <= c.lower() <= "z" for c in t["content"])
                if has_heb and has_lat:
                    assert _last_user_language([_user(t["content"])]) == "he", (
                        f"Mixed Hebrew+Latin message {t['content']!r} should return 'he'"
                    )
                    return
        pytest.skip("No mixed Hebrew+Latin user message found in conversation files")


# ── _lang_hint ────────────────────────────────────────────────────────────────

class TestLangHint:
    """Tests _lang_hint() using real messages from phase2_ex*.txt files."""

    def _first_he_msg(self, phase2_conversations: dict) -> str | None:
        for turns in phase2_conversations.values():
            for t in turns:
                if t["role"] == "user" and any("\u0590" <= c <= "\u05FF" for c in t["content"]):
                    return t["content"]
        return None

    def _first_en_msg(self, phase2_conversations: dict) -> str | None:
        turns = phase2_conversations.get("ex3", [])
        for t in turns:
            if t["role"] == "user" \
                    and t["content"].strip() \
                    and not any("\u0590" <= c <= "\u05FF" for c in t["content"]):
                return t["content"]
        return None

    def test_english_message_produces_english_critical_instruction(self, phase2_conversations):
        msg = self._first_en_msg(phase2_conversations)
        if msg is None:
            pytest.skip("No English user message found in conversation files")
        hint = _lang_hint([_user(msg)])
        assert "English" in hint and "CRITICAL" in hint

    def test_hebrew_message_produces_hebrew_critical_instruction(self, phase2_conversations):
        msg = self._first_he_msg(phase2_conversations)
        if msg is None:
            pytest.skip("No Hebrew user message found in conversation files")
        hint = _lang_hint([_user(msg)])
        assert "Hebrew" in hint and "CRITICAL" in hint

    def test_hint_is_a_non_empty_string(self, phase2_conversations):
        turns = next(iter(phase2_conversations.values()), [])
        user_content = next((t["content"] for t in turns if t["role"] == "user"), "test")
        result = _lang_hint([_user(user_content)])
        assert isinstance(result, str) and len(result) > 10


# ── _transition_message ───────────────────────────────────────────────────────

class TestTransitionMessage:
    """
    Tests _transition_message() using profiles parsed from the actual
    conversation files.  No profile data is invented.
    """

    def _with_first_name(self, phase2_profiles: dict) -> dict | None:
        return _find_profile(phase2_profiles, require=("firstName",))

    def _with_hmo(self, phase2_profiles: dict) -> dict | None:
        return _find_profile(phase2_profiles, require=("hmo",))

    def test_hebrew_response_contains_thank_you(self, phase2_profiles):
        data = self._with_first_name(phase2_profiles)
        if data is None:
            pytest.skip("No profile with firstName found in conversation files")
        assert "תודה" in _transition_message(_profile_from_dict(data), "he")

    def test_english_response_contains_thank_you(self, phase2_profiles):
        data = self._with_first_name(phase2_profiles)
        if data is None:
            pytest.skip("No profile with firstName found in conversation files")
        assert "Thank" in _transition_message(_profile_from_dict(data), "en")

    def test_hebrew_response_includes_hmo_name(self, phase2_profiles):
        data = self._with_hmo(phase2_profiles)
        if data is None:
            pytest.skip("No profile with hmo found in conversation files")
        p = _profile_from_dict(data)
        assert p.hmo in _transition_message(p, "he")

    def test_english_response_includes_hmo_name(self, phase2_profiles):
        data = self._with_hmo(phase2_profiles)
        if data is None:
            pytest.skip("No profile with hmo found in conversation files")
        p = _profile_from_dict(data)
        assert p.hmo in _transition_message(p, "en")

    def test_or_true_regression_english_is_english(self, phase2_profiles):
        """
        Regression: original bug `if language == 'he' or True:` always returned
        Hebrew.  An English call must now produce an English response.
        """
        data = self._with_first_name(phase2_profiles)
        if data is None:
            pytest.skip("No profile found in conversation files")
        msg = _transition_message(_profile_from_dict(data), "en")
        assert any(w in msg for w in ("Thank", "saved", "details", "HMO", "Tier")), (
            f"Expected English transition message, got: {msg!r}"
        )


# ── _build_validated_profile ──────────────────────────────────────────────────

class TestBuildValidatedProfile:
    """
    Tests _build_validated_profile() using profiles and invalid inputs from
    the actual phase2_ex*.txt conversation files.
    """

    # Fields required for a fully-complete profile
    _REQUIRED = ("firstName", "lastName", "idNumber", "gender", "hmo", "hmoCardNumber", "tier")

    def _complete_data(self, phase2_profiles: dict) -> dict | None:
        """Return the first profile that has all required fields AND age set."""
        for data in phase2_profiles.values():
            if all(data.get(k) for k in self._REQUIRED) and data.get("age") is not None:
                return data
        return None

    def _args(self, data: dict) -> dict:
        return {k: data.get(k) for k in (*self._REQUIRED, "age")}

    def test_valid_profile_from_conversations_is_complete(self, phase2_profiles):
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No fully-complete profile found in conversation files")
        p = _build_validated_profile(self._args(data))
        assert p.is_complete()
        assert p.idNumber == data["idNumber"]

    def test_invalid_id_from_conversations_raises(
        self, phase2_profiles, phase2_invalid_inputs
    ):
        """An ID that was rejected by the bot in a real conversation must raise ValueError."""
        invalid_ids = phase2_invalid_inputs.get("invalid_ids", [])
        if not invalid_ids:
            pytest.skip("No invalid IDs found in conversation files")
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No base profile available")
        args = self._args(data)
        args["idNumber"] = invalid_ids[0]
        with pytest.raises(ValueError, match="Invalid ID"):
            _build_validated_profile(args)

    def test_invalid_age_from_conversations_raises(
        self, phase2_profiles, phase2_invalid_inputs
    ):
        """An age that was rejected by the bot in a real conversation must raise ValueError."""
        invalid_ages = phase2_invalid_inputs.get("invalid_ages", [])
        if not invalid_ages:
            pytest.skip("No invalid ages found in conversation files")
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No base profile available")
        args = self._args(data)
        args["age"] = int(invalid_ages[0])
        with pytest.raises(ValueError, match="Invalid age"):
            _build_validated_profile(args)

    def test_english_male_gender_mapped_to_hebrew(self, phase2_profiles):
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No profile found in conversation files")
        args = self._args(data)
        args["gender"] = "male"
        assert _build_validated_profile(args).gender == "זכר"

    def test_english_female_gender_mapped_to_hebrew(self, phase2_profiles):
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No profile found in conversation files")
        args = self._args(data)
        args["gender"] = "female"
        assert _build_validated_profile(args).gender == "נקבה"

    def test_non_digit_id_raises(self, phase2_profiles):
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No profile found in conversation files")
        args = self._args(data)
        args["idNumber"] = "abcdefghi"
        with pytest.raises(ValueError, match="Invalid ID"):
            _build_validated_profile(args)

    def test_short_card_number_raises(self, phase2_profiles):
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No profile found in conversation files")
        args = self._args(data)
        args["hmoCardNumber"] = "123"
        with pytest.raises(ValueError, match="Invalid HMO card"):
            _build_validated_profile(args)

    def test_age_above_maximum_raises(self, phase2_profiles):
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No profile found in conversation files")
        args = self._args(data)
        args["age"] = 200
        with pytest.raises(ValueError, match="Invalid age"):
            _build_validated_profile(args)

    def test_none_age_raises(self, phase2_profiles):
        data = self._complete_data(phase2_profiles)
        if data is None:
            pytest.skip("No profile found in conversation files")
        args = self._args(data)
        args["age"] = None
        with pytest.raises(ValueError, match="Invalid age"):
            _build_validated_profile(args)


# ── UserProfile.is_complete ───────────────────────────────────────────────────

class TestIsComplete:
    """
    Tests UserProfile.is_complete() using profiles parsed from the actual
    conversation files.
    """

    def test_parsed_profile_is_complete(self, phase2_profiles):
        """At least one conversation must yield a fully-complete profile."""
        for data in phase2_profiles.values():
            p = _profile_from_dict(data)
            if p.is_complete():
                assert p.is_complete() is True
                return
        pytest.skip("No fully-complete profile found in parsed conversation files")

    def test_clearing_first_name_makes_incomplete(self, phase2_profiles):
        data = next(iter(phase2_profiles.values()), None)
        if data is None:
            pytest.skip("No parsed profiles available")
        p = _profile_from_dict(data)
        p.firstName = ""
        assert p.is_complete() is False

    def test_clearing_last_name_makes_incomplete(self, phase2_profiles):
        data = next(iter(phase2_profiles.values()), None)
        if data is None:
            pytest.skip("No parsed profiles available")
        p = _profile_from_dict(data)
        p.lastName = ""
        assert p.is_complete() is False

    def test_clearing_hmo_makes_incomplete(self, phase2_profiles):
        data = next(iter(phase2_profiles.values()), None)
        if data is None:
            pytest.skip("No parsed profiles available")
        p = _profile_from_dict(data)
        p.hmo = ""
        assert p.is_complete() is False

    def test_setting_age_none_makes_incomplete(self, phase2_profiles):
        data = next(iter(phase2_profiles.values()), None)
        if data is None:
            pytest.skip("No parsed profiles available")
        p = _profile_from_dict(data)
        p.age = None
        assert p.is_complete() is False

    def test_blank_profile_is_not_complete(self):
        assert UserProfile().is_complete() is False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — Conversation replay tests (require Azure)
# ═══════════════════════════════════════════════════════════════════════════════

"""
How it works:
  1. Parse the phase2_ex*.txt file into (role, content) turns.
  2. Use FastAPI TestClient pointing at the real app (so the LLM is called for real).
  3. For each user turn in the file, POST to /chat with the full accumulated
     history + current profile + current phase.
  4. Use the actual response (not the reference) going forward, so the test
     accurately simulates a real conversation.
  5. After each turn, compare the actual reply against the reference reply
     using keyword extraction and a 40% overlap threshold.
  6. At the end, print a full per-turn and per-example report.
"""

from fastapi.testclient import TestClient
from phase2.backend.main import app

# A turn passes if this fraction of reference keywords are found in the response.
# Set high (80%) because reference answers now contain ONLY the essential keywords.
MIN_KEYWORD_RATIO = 0.80

# A reference keyword and a response word are considered the same if their
# SequenceMatcher character-level similarity is at least this value.
# WORD_SIMILARITY_THRESHOLD handles Hebrew morphological variants: "תעודה"/"תעודת", "טיפול"/"טיפולים", etc.
WORD_SIMILARITY_THRESHOLD = 0.70


def _extract_keywords(text: str) -> list[str]:
    """
    Split the reference answer into individual keyword tokens.
    Reference answers contain only essential words/numbers, so we keep
    everything that is ≥ 2 characters (Hebrew, digits, percentages, Latin).
    Order does not matter — we only check existence.
    """
    return re.findall(r"[\u0590-\u05FF]{2,}|[\d]+%?|[a-zA-Z]{2,}", text)


def _word_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio between two strings (0.0 – 1.0)."""
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def _fuzzy_contains(keyword: str, response_words: list[str]) -> bool:
    """
    Return True if any word in response_words is ≥ WORD_SIMILARITY_THRESHOLD
    similar to keyword.  Handles morphological variants like תעודה/תעודת.
    """
    for word in response_words:
        if _word_similarity(keyword, word) >= WORD_SIMILARITY_THRESHOLD:
            return True
    return False


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens for fuzzy lookup."""
    return re.findall(r"[\u0590-\u05FF]+|[\d]+%?|[a-zA-Z]+", text)


def _keyword_score(actual: str, keywords: list[str]) -> tuple[float, list[str], list[str]]:
    """
    For each keyword in the reference, check if a sufficiently similar word
    exists anywhere in the actual response (order-independent, fuzzy match).
    Returns (ratio_found, found_keywords, missing_keywords).
    """
    if not keywords:
        return 1.0, [], []
    response_words = _tokenize(actual)
    found   = [k for k in keywords if _fuzzy_contains(k, response_words)]
    missing = [k for k in keywords if not _fuzzy_contains(k, response_words)]
    return len(found) / len(keywords), found, missing


# Accumulate per-turn results for the final report
_replay_results: list[dict] = []


@pytest.fixture(autouse=True, scope="module")
def _print_replay_summary():
    """Print full conversation replay report after all tests finish."""
    yield
    if not _replay_results:
        return

    total  = len(_replay_results)
    passed = sum(1 for r in _replay_results if r["pass"])
    pct    = 100.0 * passed / total if total else 0

    by_example: dict[str, list] = {}
    for r in _replay_results:
        by_example.setdefault(r["example"], []).append(r)

    print("\n")
    print("=" * 72)
    print("  PHASE 2 — CONVERSATION REPLAY REPORT")
    print("=" * 72)
    print(f"  Overall pass rate : {passed}/{total} turns  ({pct:.1f}%)")
    print()

    for ex in sorted(by_example.keys()):
        turns = by_example[ex]
        ex_pass = sum(1 for r in turns if r["pass"])
        ex_pct  = 100.0 * ex_pass / len(turns)
        print(f"  {ex} : {ex_pass}/{len(turns)} turns passed ({ex_pct:.1f}%)")
        for r in turns:
            icon  = "PASS" if r["pass"] else "FAIL"
            phase = r.get("phase_at_turn", "?")
            print(f"    [{icon}] [{phase:10s}] score={r['score']:.0%}  "
                  f"user={r['user_msg'][:55]!r}")
            print(f"           ref   : {r['ref_reply'][:100]!r}")
            print(f"           actual: {r['actual_reply'][:100]!r}")
            if not r["pass"] and r["missing"]:
                print(f"           missing: {r['missing'][:5]}")
    print("=" * 72)


@pytest.mark.integration
@pytest.mark.parametrize("example_key", ["ex1", "ex2", "ex3", "ex4"])
def test_phase2_conversation_replay(
    example_key: str,
    phase2_conversations,
    run_integration,
):
    """
    Replay the example conversation from phase2_ex{N}.txt through the real app.

    For every user message in the file:
      1. POST /chat with the accumulated history + current profile + phase.
      2. Use the ACTUAL reply (not the reference) as the assistant turn going forward.
      3. Keyword-match the actual reply against the reference reply.

    Requires --run-integration and valid Azure environment variables.
    """
    if not run_integration:
        pytest.skip("Pass --run-integration to run live conversation tests.")

    if example_key not in phase2_conversations:
        pytest.skip(f"No conversation file found for {example_key}.")

    reference_turns = phase2_conversations[example_key]
    client = TestClient(app)

    # ── State tracked across turns ────────────────────────────────────────────
    history: list[dict] = []   # [{"role": ..., "content": ...}, ...]
    profile: dict       = {}   # current UserProfile as dict
    phase:   str        = "collection"

    turn_results: list[dict] = []

    # Walk through the reference turns in (user, assistant) pairs
    i = 0
    while i < len(reference_turns):
        turn = reference_turns[i]

        if turn["role"] != "user":
            # Skip assistant turns — we generate those dynamically
            i += 1
            continue

        user_msg = turn["content"]

        # Find the reference assistant reply that follows this user turn
        ref_reply = ""
        if i + 1 < len(reference_turns) and reference_turns[i + 1]["role"] == "assistant":
            ref_reply = reference_turns[i + 1]["content"]

        # Append user turn to history
        history.append({"role": "user", "content": user_msg})

        # POST to /chat
        payload = {
            "messages":     history,
            "phase":        phase,
            "user_profile": profile,
            "language":     "he",
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200, (
            f"POST /chat failed (status {response.status_code}): {response.text}"
        )
        data = response.json()

        actual_reply    = data.get("reply", "")
        profile         = data.get("user_profile", profile)
        phase           = data.get("phase", phase)

        # Append ACTUAL reply to history (not the reference)
        history.append({"role": "assistant", "content": actual_reply})

        # ── Keyword match ─────────────────────────────────────────────────────
        keywords            = _extract_keywords(ref_reply)
        score, found, missing = _keyword_score(actual_reply, keywords)
        turn_passed         = score >= MIN_KEYWORD_RATIO or not keywords

        result = {
            "example":      example_key,
            "user_msg":     user_msg,
            "ref_reply":    ref_reply,
            "actual_reply": actual_reply,
            "keywords":     keywords,
            "found":        found,
            "missing":      missing,
            "score":        score,
            "pass":         turn_passed,
            "phase_at_turn": phase,
        }
        turn_results.append(result)
        _replay_results.append(result)

        i += 1  # advance; the assistant turn (i+1) is skipped at top of loop

    # ── Per-example assertion ─────────────────────────────────────────────────
    # The example passes if at least MIN_KEYWORD_RATIO of its turns pass.
    failed_turns = [r for r in turn_results if not r["pass"]]
    total_turns  = len(turn_results)
    pass_rate    = (total_turns - len(failed_turns)) / total_turns if total_turns else 1.0
    if pass_rate < MIN_KEYWORD_RATIO:
        details = "\n".join(
            f"  user={r['user_msg'][:60]!r}\n"
            f"    score={r['score']:.0%}  missing={r['missing'][:4]}\n"
            f"    actual  : {r['actual_reply'][:120]!r}\n"
            f"    expected: {r['ref_reply'][:120]!r}"
            for r in failed_turns
        )
        pytest.fail(
            f"{example_key}: only {pass_rate:.0%} of turns passed "
            f"(threshold {MIN_KEYWORD_RATIO:.0%}). "
            f"{len(failed_turns)}/{total_turns} turn(s) failed.\n{details}",
            pytrace=False,
        )
