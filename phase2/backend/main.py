"""
Phase 2 — FastAPI Stateless Microservice
Handles both phases:
  • collection — LLM-driven user profile collection with function calling
  • qa         — RAG-powered Q&A scoped to the user's HMO + tier

Every request is fully self-contained: the client sends the full conversation
history and user profile. No session state is stored on the server.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from openai import AzureOpenAI, APIError, APIConnectionError, RateLimitError

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent          # phase2/backend/
_PHASE2_DIR  = _HERE.parent                            # phase2/
_PROJECT_ROOT = _PHASE2_DIR.parent                     # project root
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from phase2.backend.models import (
    ChatRequest, ChatResponse, Phase, UserProfile
)
from phase2.backend.knowledge_base import get_knowledge_base

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("phase2.backend")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Knowledge base is initialised lazily on the first QA request,
    # so the server starts immediately and collection phase has zero KB overhead.
    logger.info("Phase 2 backend ready (knowledge base will load on first QA request).")
    yield
    logger.info("Shutting down Phase 2 backend.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Services ChatBot — Phase 2",
    description="Stateless microservice for HMO Q&A with RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _az_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


# ── COLLECTION PHASE TOOLS ────────────────────────────────────────────────────

COLLECTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_user_profile",
            "description": (
                "Call this function when you have collected and confirmed ALL required user information. "
                "All fields must be non-empty and validated before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "firstName":     {"type": "string", "description": "User's first name"},
                    "lastName":      {"type": "string", "description": "User's last name"},
                    "idNumber":      {"type": "string", "description": "9-digit Israeli ID number"},
                    "gender":        {"type": "string", "enum": ["זכר", "נקבה", "male", "female"], "description": "Gender"},
                    "age":           {"type": "integer", "minimum": 0, "maximum": 120, "description": "Age in years"},
                    "hmo":           {"type": "string", "enum": ["מכבי", "מאוחדת", "כללית"], "description": "HMO name"},
                    "hmoCardNumber": {"type": "string", "description": "9-digit HMO membership card number"},
                    "tier":          {"type": "string", "enum": ["זהב", "כסף", "ארד"], "description": "Insurance membership tier"},
                },
                "required": ["firstName", "lastName", "idNumber", "gender", "age", "hmo", "hmoCardNumber", "tier"],
            },
        },
    }
]

COLLECTION_SYSTEM = """\
You are a friendly, professional assistant for Israeli HMO (קופת חולים) medical services.
Your task is to collect the user's information before answering questions about their benefits.

You must collect ALL of the following, in a natural conversational way:
1. שם פרטי ושם משפחה (first and last name)
2. מספר זהות (Israeli ID — must be exactly 9 digits)
3. מין (gender: זכר/נקבה)
4. גיל (age: 0–120)
5. שם קופת החולים (HMO: מכבי / מאוחדת / כללית)
6. מספר כרטיס קופת חולים (HMO card number — must be exactly 9 digits)
7. מסלול ביטוח (insurance tier: זהב / כסף / ארד)

Rules:
- Respond in the same language the user writes in (Hebrew or English).
- Ask for missing information naturally — do not fire all questions at once.
- Validate: ID and HMO card number must be exactly 9 digits. Age must be 0–120.
  If invalid, politely ask the user to correct it.
- Before finalising, show a summary and ask the user to confirm.
  Only call `submit_user_profile` after the user explicitly confirms everything is correct.
- NEVER INFER OR ASSUME ANY FIELD VALUE. Every single field (including gender, age, HMO,
  tier, ID, card number) MUST come from an explicit answer the user provided in this
  conversation. Do NOT derive gender from the user's name, language, or any other signal.
  Do NOT assume age, HMO, or tier from anything the user said indirectly.
  If a field has not been explicitly stated by the user → you must ask for it.
- GENDER-NEUTRAL LANGUAGE: Always use impersonal/neutral Hebrew phrasing throughout the
  entire conversation. Avoid gendered verb forms (תוכלי/תוכל, תצטרכי/תצטרך, את/אתה צריך/צריכה).
  Use impersonal constructions instead: "ניתן", "אפשר", "יש צורך", "כדאי לפנות", etc.
"""

# ── Q&A PHASE PROMPT ──────────────────────────────────────────────────────────

QA_SYSTEM_TEMPLATE = """\
You are a knowledgeable and friendly medical-services assistant for Israeli health funds.

The user's verified profile:
  שם: {first_name} {last_name}
  קופת חולים: {hmo}
  מסלול ביטוח: {tier}

Your role is to answer questions about medical services and benefits available to this user
based ONLY on the provided knowledge-base context below.

STRICT RULES:
1. Answer ONLY about services for {hmo} at the {tier} tier.
   Do NOT mention benefits for other HMOs or tiers.
2. Base your answers STRICTLY on the context provided. Do NOT invent, infer, or elaborate
   beyond what is explicitly written in the context — no invented steps, procedures, or
   contact details. If a phone number or website is in the context, quote it exactly.
   If the information is not in the context, say: "אין לי מידע על כך בקופת החולים שלך."
3. Respond in the SAME LANGUAGE the user writes in (Hebrew or English).
4. Be concise, accurate, and friendly.
   Always use impersonal/neutral Hebrew phrasing — replace gendered verbs (תוכלי/תוכל,
   תצטרכי/תצטרך) with neutral forms: "ניתן", "אפשר", "יש צורך", "כדאי לפנות", etc.
5. If asked to compare with other HMOs/tiers, politely decline — you can only discuss the user's plan.
6. Think semantically: if the user asks about a topic (e.g., "חרדות", "לחץ"), consider whether
   any service in the context addresses a related or synonymous need (e.g., stress management,
   relaxation workshops) and offer it as a relevant option rather than saying there is no information.
7. For follow-up questions (e.g., "איך נרשמים?"), use the full conversation history to infer
   what service the user is referring to, and answer accordingly.

KNOWLEDGE BASE CONTEXT (relevant to {hmo} — {tier}):
---
{context}
---
"""


# ── Core LLM call ─────────────────────────────────────────────────────────────

def _call_llm(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> dict:
    """
    Unified LLM call with retry logic.
    Returns the raw choice object as a dict.
    """
    client     = _az_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT4O", "gpt-4o")

    kwargs: dict = dict(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            choice   = response.choices[0]
            return {
                "finish_reason":  choice.finish_reason,
                "content":        choice.message.content or "",
                "tool_calls":     choice.message.tool_calls,
            }
        except RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning("Rate limit (attempt %d/%d). Sleeping %ds.", attempt, max_retries, wait)
            time.sleep(wait)
            last_exc = exc
        except (APIConnectionError, APIError) as exc:
            logger.error("Azure OpenAI error: %s", exc)
            last_exc = exc
            break

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_exc}")


# ── Phase handlers ─────────────────────────────────────────────────────────────

def _handle_collection(req: ChatRequest) -> ChatResponse:
    """
    Drive user-profile collection via function calling.
    Returns updated profile and transitions to QA phase when complete.
    """
    system = COLLECTION_SYSTEM + _lang_hint(req.messages)
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    result = _call_llm(messages, tools=COLLECTION_TOOLS, temperature=0.4)

    # Check if the LLM called submit_user_profile
    if result["tool_calls"]:
        for tc in result["tool_calls"]:
            if tc.function.name == "submit_user_profile":
                try:
                    args = json.loads(tc.function.arguments)
                    profile = _build_validated_profile(args)
                    if profile.is_complete():
                        # Transition to QA phase
                        lang = _last_user_language(req.messages)
                        return ChatResponse(
                            reply=_transition_message(profile, lang),
                            user_profile=profile,
                            phase=Phase.qa,
                        )
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Profile submission parsing error: %s", exc)

    # Still in collection phase
    return ChatResponse(
        reply=result["content"],
        user_profile=req.user_profile,
        phase=Phase.collection,
    )


def _handle_qa(req: ChatRequest) -> ChatResponse:
    """Answer user questions using RAG filtered to their HMO + tier."""
    profile = req.user_profile
    if not profile.is_complete():
        return ChatResponse(
            reply="לא נמצאו פרטי משתמש מלאים. אנא התחל מחדש.",
            user_profile=profile,
            phase=Phase.collection,
        )

    # Build RAG query from last 2 user turns so follow-up questions carry context
    user_messages = [m for m in req.messages if m.role == "user"]
    query = " ".join(m.content for m in user_messages[-2:]) if user_messages else ""

    # Retrieve relevant context
    kb = get_knowledge_base()
    try:
        chunks = kb.retrieve(query, hmo=profile.hmo, tier=profile.tier, top_k=10)
        # Always append the contact chunk for the top result's category so
        # registration/phone questions are never answered with "אין מידע"
        if chunks:
            contact = kb.get_contact_chunk(profile.hmo, profile.tier, chunks[0].category)
            if contact and contact not in chunks:
                chunks.append(contact)
        context = "\n\n---\n\n".join(c.content for c in chunks)
    except Exception as exc:
        logger.error("Knowledge base retrieval error: %s", exc)
        context = kb.get_full_context_for_user(profile.hmo, profile.tier)

    system_msg = QA_SYSTEM_TEMPLATE.format(
        first_name=profile.firstName,
        last_name=profile.lastName,
        hmo=profile.hmo,
        tier=profile.tier,
        context=context,
    )

    system_msg += _lang_hint(req.messages)
    messages = [{"role": "system", "content": system_msg}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    result = _call_llm(messages, temperature=0.3, max_tokens=1024)

    return ChatResponse(
        reply=result["content"],
        user_profile=profile,
        phase=Phase.qa,
    )


def _build_validated_profile(args: dict) -> UserProfile:
    """Build and validate UserProfile from function-call arguments."""
    id_num     = str(args.get("idNumber", "")).strip()
    card_num   = str(args.get("hmoCardNumber", "")).strip()
    age        = args.get("age")

    if not (id_num.isdigit() and len(id_num) == 9):
        raise ValueError(f"Invalid ID number: {id_num}")
    if not (card_num.isdigit() and len(card_num) == 9):
        raise ValueError(f"Invalid HMO card number: {card_num}")
    if age is None or not (0 <= int(age) <= 120):
        raise ValueError(f"Invalid age: {age}")

    gender_map = {"male": "זכר", "female": "נקבה"}
    gender = args.get("gender", "")
    gender = gender_map.get(gender.lower(), gender)

    return UserProfile(
        firstName=args.get("firstName", ""),
        lastName=args.get("lastName", ""),
        idNumber=id_num,
        gender=gender,
        age=int(age),
        hmo=args.get("hmo", ""),
        hmoCardNumber=card_num,
        tier=args.get("tier", ""),
    )


def _last_user_language(messages: list) -> str:
    """Return 'he' if the last user message contains Hebrew, else 'en'."""
    for m in reversed(messages):
        if m.role == "user" and m.content:
            if any("\u0590" <= c <= "\u05FF" for c in m.content):
                return "he"
            return "en"
    return "he"


def _lang_hint(messages: list) -> str:
    """Return an explicit language instruction to append to the system prompt."""
    lang = _last_user_language(messages)
    if lang == "en":
        return "\n\nCRITICAL: The user's last message is in English. You MUST reply in English only."
    return "\n\nCRITICAL: The user's last message is in Hebrew. You MUST reply in Hebrew only."


def _transition_message(profile: UserProfile, language: str) -> str:
    if language == "en":
        return (
            f"Thank you, {profile.firstName}! Your details have been saved.\n\n"
            f"**HMO:** {profile.hmo} | **Tier:** {profile.tier}\n\n"
            "I can now answer questions about your medical services. What would you like to know? 😊"
        )
    return (
        f"תודה {profile.firstName}! הפרטים שלך נשמרו בהצלחה.\n\n"
        f"**קופת חולים:** {profile.hmo} | **מסלול:** {profile.tier}\n\n"
        "כעת אני יכול לענות על שאלות לגבי השירותים הרפואיים שלך. "
        "מה תרצה לדעת? 😊"
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    try:
        kb = get_knowledge_base()
        chunk_count = len(kb._chunks)
    except RuntimeError:
        chunk_count = 0
    return {
        "status": "healthy",
        "knowledge_base_chunks": chunk_count,
        "version": "1.0.0",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    logger.info(
        "Chat request | phase=%s | hmo=%s | tier=%s | messages=%d",
        req.phase, req.user_profile.hmo, req.user_profile.tier, len(req.messages),
    )
    try:
        if req.phase == Phase.collection:
            return _handle_collection(req)
        else:
            return _handle_qa(req)
    except RuntimeError as exc:
        logger.error("Chat handler error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": "An unexpected error occurred."},
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "phase2.backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info",
    )
