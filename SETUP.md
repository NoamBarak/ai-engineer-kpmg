# Setup & Run Instructions

## Prerequisites

- Python 3.10+
- Azure credentials (Document Intelligence + Azure OpenAI) provided in the assignment email

---

## 1. Environment Setup

```bash
# Clone / navigate to the project root
cd Home-Assignment-GenAI-KPMG-...

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## 2. Configure Azure Credentials

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and fill in your Azure credentials:

```ini
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://YOUR-RESOURCE.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=YOUR_KEY

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_KEY=YOUR_KEY
AZURE_OPENAI_API_VERSION=2024-02-01

# Deployment names (must match what's deployed in your Azure portal)
AZURE_OPENAI_DEPLOYMENT_GPT4O=gpt-4o
AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI=gpt-4o-mini
AZURE_OPENAI_DEPLOYMENT_ADA002=text-embedding-ada-002

# Phase 2
PHASE2_BACKEND_URL=http://localhost:8000
PHASE2_DATA_PATH=./phase2_data
```

---

## 3. Run Phase 1 — Form 283 Extractor

```bash
# From the project root
streamlit run phase1/app.py
```

Opens at **http://localhost:8501**

**Usage:**
1. Upload any of the `phase1_data/*.pdf` files (or a JPG/PNG image)
2. The app runs OCR via Azure Document Intelligence
3. GPT-4o maps the extracted text to the JSON schema
4. View the result in the "Raw JSON" or "Field-by-Field View" tabs
5. Download the extracted JSON with the download button

---

## 4. Run Phase 2 — Medical Services ChatBot

Phase 2 requires two processes running simultaneously.

### Terminal 1 — Start the FastAPI backend

```bash
# From the project root
python -m uvicorn phase2.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Or equivalently:

```bash
python phase2/backend/main.py
```

The backend will:
1. Parse all 6 HTML files in `phase2_data/`
2. Embed ~108 knowledge chunks using ADA-002 (takes ~30 seconds on first run)
3. Serve at **http://localhost:8000**

Health check: http://localhost:8000/health

### Terminal 2 — Start the Gradio frontend

```bash
# From the project root
python phase2/frontend.py
```

Opens at **http://localhost:7860**

**Usage:**
1. The bot starts the **Information Collection** phase automatically
2. Answer the bot's questions conversationally (name, ID, HMO, tier, etc.)
3. Confirm your details when asked
4. The bot transitions to the **Q&A** phase
5. Ask questions about your medical services — answers are scoped to your HMO and tier

**Example questions:**
- "כמה טיפולי דיקור סיני אני יכול לקבל בשנה?"
- "מה ההנחה שלי על ניתוח לייזר?"
- "מה כלול במסלול הזהב שלי בנושא שיניים?"
- "How many acupuncture treatments do I get per year?"

---

## Project Structure

```
├── phase1/
│   ├── app.py              # Streamlit UI for Form 283 extraction
│   ├── ocr_processor.py    # Azure Document Intelligence wrapper
│   └── llm_extractor.py    # GPT-4o extraction prompt & logic
├── phase2/
│   ├── frontend.py         # Gradio UI (client-side state management)
│   └── backend/
│       ├── main.py         # FastAPI stateless microservice
│       ├── knowledge_base.py  # HTML parsing + ADA-002 embeddings + retrieval
│       └── models.py       # Pydantic request/response models
├── phase1_data/            # Form 283 PDF examples
├── phase2_data/            # HTML knowledge base files
├── requirements.txt
├── .env.example
└── SETUP.md
```

---

## Architecture

### Phase 1

```
Streamlit UI
    ↓ (file bytes)
Azure Document Intelligence (prebuilt-layout model)
    ↓ (structured OCR text with page layout)
GPT-4o (extraction prompt with JSON mode)
    ↓
Structured JSON (Form 283 schema)
    ↓
Streamlit UI (display + download)
```

### Phase 2

```
Gradio UI (owns all state: history + user profile)
    ↓ (full history + profile + phase on every request)
FastAPI (stateless — no session memory)
    ├── Collection phase:  GPT-4o + Function Calling → validate → collect profile
    └── Q&A phase:         ADA-002 embed query → cosine similarity on HMO+tier filtered chunks
                           → GPT-4o answer generation with retrieved context
    ↓
ChatResponse (reply + updated profile + phase)
    ↓
Gradio UI (update state + display)
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing AZURE_OPENAI_ENDPOINT` | Fill in `.env` correctly |
| Backend starts but knowledge base fails | Check `PHASE2_DATA_PATH` points to `phase2_data/` |
| `Rate limit` errors | The ADA-002 embedding call hits limits. Wait a few seconds and retry |
| Gradio shows "Cannot connect to backend" | Start the FastAPI server first (Terminal 1) |
| OCR returns empty text | Ensure the PDF is not password-protected |
| `azure-ai-documentintelligence` not found | Run `pip install azure-ai-documentintelligence` |
