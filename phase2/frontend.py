"""
Phase 2 — Gradio Frontend

All conversation history and user profile are managed client-side via gr.State.
Every message sends the full state to the FastAPI backend; no session memory
is kept on the server.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("PHASE2_BACKEND_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
REQUEST_TIMEOUT = 60  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("phase2.frontend")

# ── Initial greeting ──────────────────────────────────────────────────────────
WELCOME_MESSAGE = (
    "שלום! 👋 אני הבוט הרפואי של קופות החולים.\n\n"
    "אני כאן לעזור לך למצוא מידע על שירותים רפואיים ב**מכבי**, **מאוחדת** ו**כללית**.\n\n"
    "נתחיל בכמה פרטים קצרים כדי שאוכל לענות על שאלות ספציפיות עבורך.\n\n"
    "**מה שמך?** 😊\n\n"
    "---\n"
    "*Hello! I'm the medical services chatbot. I can help you find information about HMO benefits. "
    "Let's start with a few details — what's your name?*"
)

# Initial chatbot history in Gradio 6 messages format
INIT_HISTORY = [{"role": "assistant", "content": WELCOME_MESSAGE}]


# ── API helpers ───────────────────────────────────────────────────────────────

def _call_backend(messages: list[dict], user_profile: dict, phase: str) -> dict:
    """Send a request to the FastAPI backend and return the parsed response."""
    payload = {
        "messages": messages,
        "user_profile": user_profile,
        "phase": phase,
    }
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.post(CHAT_ENDPOINT, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        return {
            "reply": "⏱️ הבקשה ארכה יותר מדי. אנא נסה שוב. / Request timed out. Please try again.",
            "user_profile": user_profile,
            "phase": phase,
            "error": "timeout",
        }
    except httpx.HTTPStatusError as exc:
        msg = f"שגיאה בשרת (קוד {exc.response.status_code}). / Server error ({exc.response.status_code})."
        logger.error("Backend HTTP error: %s — %s", exc.response.status_code, exc.response.text)
        return {"reply": msg, "user_profile": user_profile, "phase": phase, "error": str(exc)}
    except httpx.RequestError as exc:
        msg = (
            "לא ניתן להתחבר לשרת. ודא שה-backend פועל. / "
            "Cannot connect to backend. Make sure the FastAPI server is running."
        )
        logger.error("Connection error: %s", exc)
        return {"reply": msg, "user_profile": user_profile, "phase": phase, "error": str(exc)}


def _check_backend() -> tuple[bool, str]:
    """Ping the health endpoint; return (is_healthy, status_message)."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(HEALTH_ENDPOINT)
            resp.raise_for_status()
            data = resp.json()
            chunks = data.get("knowledge_base_chunks", "?")
            return True, f"✅ Backend healthy — {chunks} knowledge chunks loaded"
    except Exception as exc:
        return False, f"❌ Backend unreachable: {exc}"


# ── Chat function ─────────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list,                       # Gradio 6 messages format
    messages_state: list,                # OpenAI-format messages sent to backend
    user_profile_state: dict,
    phase_state: str,
):
    """Main chat handler — called on each user submission."""
    if not user_message.strip():
        return "", history, messages_state, user_profile_state, phase_state, \
               _format_profile(user_profile_state), _phase_badge(phase_state)

    # Append user message to Gradio history and backend messages
    history = history + [{"role": "user", "content": user_message}]
    messages_state = messages_state + [{"role": "user", "content": user_message}]

    # Call backend
    response = _call_backend(messages_state, user_profile_state, phase_state)

    bot_reply   = response.get("reply", "שגיאה. / Error.")
    new_profile = response.get("user_profile", user_profile_state)
    new_phase   = response.get("phase", phase_state)

    # Append assistant reply to both histories
    history = history + [{"role": "assistant", "content": bot_reply}]
    messages_state = messages_state + [{"role": "assistant", "content": bot_reply}]

    logger.info("Phase: %s → %s | Profile complete: %s", phase_state, new_phase,
                _is_profile_complete(new_profile))

    return (
        "",
        history,
        messages_state,
        new_profile,
        new_phase,
        _format_profile(new_profile),
        _phase_badge(new_phase),
    )


def reset_chat():
    """Reset all state to initial values."""
    init_messages = []
    init_profile  = {
        "firstName": "", "lastName": "", "idNumber": "",
        "gender": "", "age": None, "hmo": "", "hmoCardNumber": "", "tier": "",
    }
    return (
        list(INIT_HISTORY),
        init_messages,
        init_profile,
        "collection",
        _format_profile(init_profile),
        _phase_badge("collection"),
        "",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_profile(profile: dict) -> str:
    if not profile or not any(profile.values()):
        return "טרם נאספו פרטים / No profile yet"
    lines = []
    field_labels = {
        "firstName": "👤 שם פרטי", "lastName": "👤 שם משפחה",
        "idNumber": "🪪 ת.ז.", "gender": "⚧ מין", "age": "🎂 גיל",
        "hmo": "🏥 קופה", "hmoCardNumber": "💳 כרטיס קופה", "tier": "⭐ מסלול",
    }
    for key, label in field_labels.items():
        val = profile.get(key)
        if val is not None and str(val).strip():
            lines.append(f"{label}: {val}")
    return "  \n".join(lines) if lines else "טרם נאספו פרטים / No profile yet"


def _is_profile_complete(profile: dict) -> bool:
    required = ["firstName", "lastName", "idNumber", "gender", "age", "hmo", "hmoCardNumber", "tier"]
    return all(profile.get(f) for f in required)


def _phase_badge(phase: str) -> str:
    if phase == "collection":
        return "🔵 שלב איסוף פרטים / Info Collection"
    return "🟢 שלב שאלות ותשובות / Q&A"


# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
#chatbot .message.bot { background: #eef4ff; border-radius: 12px; }
#chatbot .message.user { background: #f0fff4; border-radius: 12px; }
.header-box {
    background: linear-gradient(135deg, #A7C7E7 0%, #eef4ff 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1rem;
}
.phase-badge { font-size: 1.1rem; font-weight: bold; padding: 0.4rem 0.8rem; }
.profile-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.9rem;
}
"""

# ── Build Gradio UI ────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    # Initial state
    init_history   = list(INIT_HISTORY)
    init_messages  = []
    init_profile   = {
        "firstName": "", "lastName": "", "idNumber": "",
        "gender": "", "age": None, "hmo": "", "hmoCardNumber": "", "tier": "",
    }

    with gr.Blocks(title="Medical Services ChatBot") as demo:

        # ── State ──────────────────────────────────────────────────────────
        messages_state      = gr.State(init_messages)
        user_profile_state  = gr.State(init_profile)
        phase_state         = gr.State("collection")

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <h1 style="margin:0; font-size:2rem; font-family:'Poppins',sans-serif; font-weight:600; letter-spacing:-0.5px;">🏥 Medical Services ChatBot</h1>
            <p style="margin:0.5rem 0 0; opacity:0.9">
                שירותים רפואיים | מכבי · מאוחדת · כללית
            </p>
        </div>
        """)

        with gr.Row():
            # ── Chat panel ─────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=init_history,
                    elem_id="chatbot",
                    height=520,
                    show_label=False,
                )

                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="הקלד הודעה... / Type a message...",
                        show_label=False,
                        scale=9,
                        container=False,
                        lines=1,
                        autofocus=True,
                    )
                    send_btn = gr.Button("שלח / Send", scale=1, variant="primary")

                with gr.Row():
                    reset_btn = gr.Button("🔄 התחל מחדש / Reset", variant="secondary", size="sm")

            # ── Sidebar ────────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Session Status")

                phase_display = gr.Markdown(
                    value=_phase_badge("collection"),
                    elem_classes=["phase-badge"],
                )

                gr.Markdown("### 👤 User Profile")
                profile_display = gr.Markdown(
                    value=_format_profile(init_profile),
                    elem_classes=["profile-card"],
                )

                gr.Markdown("### ℹ️ How to Use")
                gr.Markdown(
                    """
                    **Phase 1 — Information Collection:**
                    The bot will ask for your details conversationally.

                    **Phase 2 — Q&A:**
                    Once your profile is set, ask any question about your HMO benefits.

                    **Example questions:**
                    - "כמה טיפולי דיקור יש לי?💉"
                    - "מה ההנחה על ניתוח לייזר?🤓"
                    - "מה כולל מסלול הזהב שלי?🪙"
                    """
                )

                # Backend status (shown at load)
                with gr.Accordion("🔌 Backend Status", open=False):
                    backend_status = gr.Markdown()
                    check_btn = gr.Button("Check Backend", size="sm")

        # ── Event wiring ───────────────────────────────────────────────────
        outputs = [
            txt, chatbot, messages_state, user_profile_state,
            phase_state, profile_display, phase_display
        ]

        txt.submit(
            fn=chat,
            inputs=[txt, chatbot, messages_state, user_profile_state, phase_state],
            outputs=outputs,
        )
        send_btn.click(
            fn=chat,
            inputs=[txt, chatbot, messages_state, user_profile_state, phase_state],
            outputs=outputs,
        )
        reset_btn.click(
            fn=reset_chat,
            inputs=[],
            outputs=[
                chatbot, messages_state, user_profile_state,
                phase_state, profile_display, phase_display, txt
            ],
        )

        def _check():
            ok, msg = _check_backend()
            return msg

        check_btn.click(fn=_check, inputs=[], outputs=[backend_status])

        # Auto-check backend on load
        demo.load(fn=_check, inputs=[], outputs=[backend_status])

    return demo


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", 7860)),
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )