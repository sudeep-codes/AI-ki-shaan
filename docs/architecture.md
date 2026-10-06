# Architecture — Agri Chatbot

## High-level flow

```
Telegram message (text or photo)
        |
        v
  Language detection
        |
        v
  [if non-English] Translate to English
        |
        v
  Intent router
   /     |      \
  /      |       \
Fertilizer  Image    Finance
 (RAG)    (CNN)    (RAG, scheme docs)
  \        |        /
   \       |       /
    Gemini generation (grounded on retrieved context)
        |
        v
  [if non-English] Translate back to user's language
        |
        v
  Reply sent via Telegram
```

## Components

### 1. Telegram interface (`/bot`)
- `python-telegram-bot` handles incoming messages (text, photo, commands)
- Commands: `/start`, `/language`, `/finance`, `/help`
- Routes text to the language/intent pipeline, routes photos to the image pipeline

### 2. Language layer (`/bot/language.py`)
- Detect language of incoming text (langdetect or Gemini itself)
- Translate non-English input to English before retrieval (retrieval quality is better in English since most source docs are English)
- Generate final response directly in the user's language (Gemini can do this in the same generation call if instructed — avoids a second translation round-trip and reads more naturally)

### 3. RAG pipeline (`/rag`)
- `ingest.py` — loads source PDFs/text from `/data`, chunks (~500 tokens, slight overlap), embeds with a local sentence-transformers model, stores in ChromaDB
- `query.py` — takes a question, returns top-k relevant passages with source doc names attached (needed for the citation rule in `rules.md`)
- Two separate ChromaDB collections: one for fertilizer/agronomy docs, one for finance/scheme docs — keeps retrieval focused and avoids the two domains polluting each other's results

### 4. Image diagnosis (`/models`)
- Pretrained plant-disease classifier (PlantVillage-based, from Hugging Face)
- Input: photo → preprocess (resize/normalize) → prediction + confidence score
- If confidence is below threshold (e.g. 0.6), the bot asks a clarifying question instead of stating a diagnosis
- Predicted disease label is used as a query into the fertilizer/agronomy RAG collection to fetch a remedy

### 5. Intent router (`/bot/router.py`)
- Simple rule-based first pass (keywords: "loan", "scheme", "insurance" → finance; photo present → image; else → fertilizer/general RAG)
- Can be upgraded later to a small classifier if keyword routing proves too brittle

### 6. Generation layer
- All final answers go through one Gemini call with: system prompt (grounding rules) + retrieved context + user question
- Temperature kept low for factual consistency

### 7. Deployment
- FastAPI app wrapping the Telegram webhook (not polling, once hosted)
- Hosted on Render free tier or Hugging Face Spaces
- Environment variables for all API keys, never committed

## Data flow notes
- No user data is persisted beyond the current session for MVP (no database) — keeps privacy simple and matches scope
- Source documents live in `/data`, versioned in the repo so retrieval is reproducible

## Tech choices and why
| Choice | Alternative considered | Why this one |
|---|---|---|
| Telegram Bot API | WhatsApp Business API | Free, no business approval process, faster to ship |
| ChromaDB (local) | Pinecone | Free, no external service dependency, fine at this scale |
| sentence-transformers (local embeddings) | OpenAI/Gemini embeddings API | Free, no per-call cost, runs offline |
| Gemini API free tier | GPT-4 API | No cost at prototype scale |
| Render/HF Spaces free tier | AWS/GCP paid hosting | Free for low-traffic prototype |
