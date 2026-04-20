"""
Phase 1 — Form 283 Extractor
Streamlit UI: upload a PDF or image → Azure Document Intelligence OCR
              → GPT-4o extraction → display structured JSON.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Support running from project root OR from within phase1/
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from phase1.ocr_processor import extract_text_from_bytes
from phase1.llm_extractor import extract_fields, validate_extraction

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("phase1.app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Form 283 — Field Extractor",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    body { direction: ltr; }
    .main-header {
        background: linear-gradient(90deg, #A7C7E7 0%, #66B2F5 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .section-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .validation-warn { color: #856404; background: #fff3cd; padding: 4px 8px; border-radius: 4px; }
    .validation-err  { color: #721c24; background: #f8d7da; padding: 4px 8px; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _content_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")


def _check_env() -> list[str]:
    missing = []
    for var in (
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "AZURE_DOCUMENT_INTELLIGENCE_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
    ):
        if not os.environ.get(var):
            missing.append(var)
    return missing


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Form 283 Extractor")
    st.markdown(
        """
        **How it works:**
        1. 📂 Upload: Form 283 (PDF or image)
        2. 🌨️ Azure Document Intelligence performs OCR
        3. 🖥️ GPT-4o maps the text to a structured JSON
        4. 📄 Review the result and validation report

        **Supported formats:** PDF, JPG, PNG, TIFF, BMP
        """
    )
    st.divider()
    show_ocr = st.checkbox("Show raw OCR text", value=False)
    st.caption("Phase 1 — GenAI Developer Assessment- KPMG")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h2 style="margin:0">📋 Form 283  — Field Extractor</h2>
        <p style="margin:0.3rem 0 0 0; opacity:0.85">
            Upload a filled Form 283 to extract all fields into structured JSON
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Env check ─────────────────────────────────────────────────────────────────
missing_vars = _check_env()
if missing_vars:
    st.error(
        "**Missing environment variables:**\n\n"
        + "\n".join(f"- `{v}`" for v in missing_vars)
        + "\n\nCopy `.env.example` to `.env` and fill in your Azure credentials."
    )
    st.stop()

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Form 283 (PDF or image)",
    type=["pdf", "jpg", "jpeg", "png", "tiff", "bmp"],
    help="Upload a completed Form 283",
)

if not uploaded:
    st.info("👆 Upload a Form 283 file to begin extraction.")
    st.stop()

# ── Process ───────────────────────────────────────────────────────────────────
file_bytes = uploaded.read()
ct = _content_type(uploaded.name)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Uploaded Document")
    if ct == "application/pdf":
        st.download_button(
            "📄 Download uploaded PDF",
            data=file_bytes,
            file_name=uploaded.name,
            mime="application/pdf",
        )
        st.info("PDF preview not supported in browser — download to view.")
    else:
        st.image(file_bytes, caption=uploaded.name, use_container_width=True)

with col_right:
    with st.spinner("🔍 Running OCR via Azure Document Intelligence..."):
        try:
            ocr_text = extract_text_from_bytes(file_bytes, ct)
            logger.info("OCR succeeded. Chars extracted: %d", len(ocr_text))
        except Exception as exc:
            st.error(f"OCR failed: {exc}")
            logger.exception("OCR error")
            st.stop()

    if show_ocr:
        with st.expander("Raw OCR Text", expanded=False):
            st.code(ocr_text, language=None)

    with st.spinner("🤖 Extracting fields with GPT-4o..."):
        try:
            extracted = extract_fields(ocr_text)
            logger.info("Field extraction complete.")
        except Exception as exc:
            st.error(f"Field extraction failed: {exc}")
            logger.exception("Extraction error")
            st.stop()

    # ── Validation report ─────────────────────────────────────────────────
    validation = validate_extraction(extracted)

    if validation["errors"]:
        for err in validation["errors"]:
            st.markdown(f'<div class="validation-err">⛔ {err}</div>', unsafe_allow_html=True)

    if validation["warnings"]:
        for warn in validation["warnings"]:
            st.markdown(f'<div class="validation-warn">⚠️ {warn}</div>', unsafe_allow_html=True)

    if not validation["errors"] and not validation["warnings"]:
        st.success("✅ Extraction complete — no validation issues detected.")

# ── Result display ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Extracted JSON")

tab_json, tab_fields = st.tabs(["📄 Raw JSON", "🗂 Field-by-Field View"])

with tab_json:
    json_str = json.dumps(extracted, ensure_ascii=False, indent=2)
    st.code(json_str, language="json")
    st.download_button(
        "⬇️ Download JSON",
        data=json_str.encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_extracted.json",
        mime="application/json",
    )

with tab_fields:
    # Personal info
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 👤 Personal Information")
        c1, c2, c3 = st.columns(3)
        c1.text_input("Last Name (שם משפחה)", value=extracted.get("lastName", ""), disabled=True)
        c2.text_input("First Name (שם פרטי)", value=extracted.get("firstName", ""), disabled=True)
        c3.text_input("ID Number (מספר זהות)", value=extracted.get("idNumber", ""), disabled=True)

        c4, c5 = st.columns(2)
        c4.text_input("Gender (מין)", value=extracted.get("gender", ""), disabled=True)

        dob = extracted.get("dateOfBirth", {})
        c5.text_input(
            "Date of Birth (תאריך לידה)",
            value=f"{dob.get('day','')}/{dob.get('month','')}/{dob.get('year','')}",
            disabled=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Address
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🏠 Address (כתובת)")
        addr = extracted.get("address", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.text_input("Street (רחוב)", value=addr.get("street", ""), disabled=True)
        c2.text_input("House No.", value=addr.get("houseNumber", ""), disabled=True)
        c3.text_input("Entrance", value=addr.get("entrance", ""), disabled=True)
        c4.text_input("Apt.", value=addr.get("apartment", ""), disabled=True)
        c5, c6, c7 = st.columns(3)
        c5.text_input("City (ישוב)", value=addr.get("city", ""), disabled=True)
        c6.text_input("Postal Code", value=addr.get("postalCode", ""), disabled=True)
        c7.text_input("P.O. Box", value=addr.get("poBox", ""), disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Contact & work
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 📞 Contact & Work")
        c1, c2, c3 = st.columns(3)
        c1.text_input("Landline (טלפון קווי)", value=extracted.get("landlinePhone", ""), disabled=True)
        c2.text_input("Mobile (נייד)", value=extracted.get("mobilePhone", ""), disabled=True)
        c3.text_input("Job Type (סוג עבודה)", value=extracted.get("jobType", ""), disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Accident info
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🚨 Accident Information")
        doi = extracted.get("dateOfInjury", {})
        c1, c2, c3 = st.columns(3)
        c1.text_input(
            "Date of Injury (תאריך פגיעה)",
            value=f"{doi.get('day','')}/{doi.get('month','')}/{doi.get('year','')}",
            disabled=True,
        )
        c2.text_input("Time of Injury (שעה)", value=extracted.get("timeOfInjury", ""), disabled=True)
        c3.text_input("Accident Location (מקום)", value=extracted.get("accidentLocation", ""), disabled=True)
        st.text_input("Accident Address", value=extracted.get("accidentAddress", ""), disabled=True)
        st.text_area(
            "Accident Description (תיאור התאונה)",
            value=extracted.get("accidentDescription", ""),
            disabled=True,
            height=80,
        )
        st.text_input("Injured Body Part (האיבר שנפגע)", value=extracted.get("injuredBodyPart", ""), disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Dates & signature
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 📅 Dates & Signature")
        ffd = extracted.get("formFillingDate", {})
        frc = extracted.get("formReceiptDateAtClinic", {})
        c1, c2, c3 = st.columns(3)
        c1.text_input("Signature (חתימה)", value=extracted.get("signature", ""), disabled=True)
        c2.text_input(
            "Form Filling Date",
            value=f"{ffd.get('day','')}/{ffd.get('month','')}/{ffd.get('year','')}",
            disabled=True,
        )
        c3.text_input(
            "Receipt Date at Clinic",
            value=f"{frc.get('day','')}/{frc.get('month','')}/{frc.get('year','')}",
            disabled=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Medical institution fields
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🏥 Medical Institution Fields (למילוי ע\"י המוסד)")
        mif = extracted.get("medicalInstitutionFields", {})
        c1, c2 = st.columns(2)
        c1.text_input("Health Fund Member (קופת חולים)", value=mif.get("healthFundMember", ""), disabled=True)
        c2.text_input("Nature of Accident (מהות התאונה)", value=mif.get("natureOfAccident", ""), disabled=True)
        st.text_area(
            "Medical Diagnoses (אבחנות רפואיות)",
            value=mif.get("medicalDiagnoses", ""),
            disabled=True,
            height=80,
        )
        st.markdown("</div>", unsafe_allow_html=True)

