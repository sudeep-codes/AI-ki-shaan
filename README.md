# 🌾 AI-ki-shan (एआई-किसान)

> **Empowering Farmers & Agronomists with AI-Driven Agricultural Intelligence**

AI-ki-shan is an intelligent agriculture backend optimized for deployment on **Hugging Face Spaces (16GB RAM environment)**. Built with **FastAPI**, **python-telegram-bot**, and a high-performance **RAG (Retrieval-Augmented Generation)** pipeline using **ChromaDB**, **Sentence-Transformers**, and **Unstructured** for table & document parsing.

---

## 🚀 Key Features

- **🌾 Agricultural Advisory Engine**: Real-time recommendations on crop disease diagnosis, integrated pest management (IPM), soil testing, fertilizer application (NPK schedules), and irrigation.
- **📜 Government Schemes Navigator**: Instant guidance on schemes like **PM-KISAN**, **PMFBY** (Crop Insurance), **Kisan Credit Card (KCC)**, and Soil Health Cards.
- **⚡ Hugging Face Spaces Ready**: Native support for port `7860`, CPU/GPU inference optimization, and lightweight memory footprint tailored for 16GB RAM containers.
- **🤖 Multi-Channel Bot Support**: Complete Telegram bot integration with `/start`, `/schemes`, `/soil`, `/pests` commands and freeform conversational understanding.
- **📊 Table & Document Ingestion**: Ingest PDF reports, CSV tables, and research bulletins via `unstructured` into a persistent **ChromaDB** vector database.
- **🔌 Multi-Provider LLM Orchestration**: Supports Hugging Face Inference API, OpenAI, Gemini, and intelligent local extractive synthesis.

---

## 📁 Project Structure

```
AI-KI-SHAN/
├── app.py                     # FastAPI application entrypoint (HF Spaces compatible)
├── config.py                  # Pydantic environment & application configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Template for environment variables
├── .gitignore                 # Git ignore rules
│
├── bot/                       # Telegram Bot Integration
│   ├── __init__.py
│   ├── bot_service.py         # Bot lifecycle manager (Polling & Webhook modes)
│   ├── chatbot_pipeline.py    # Multi-turn Gemini chatbot pipeline
│   ├── handlers.py            # Command, callback, and natural language handlers
│   └── telegram_handlers.py   # Crop disease photo diagnosis & intent handlers
│
├── finance/                   # Financial Schemes & Intent Routing
│   └── scheme_router.py       # Intent router (PM-KISAN, PMFBY, KCC vs Agriculture)
│
├── rag/                       # Retrieval-Augmented Generation
│   ├── __init__.py
│   ├── document_loader.py     # Document & table loader using 'unstructured'
│   ├── vector_store.py        # ChromaDB vector store manager
│   ├── retriever.py           # Agriculture domain semantic retriever
│   ├── pipeline.py            # End-to-end RAG query & ingestion pipeline
│   └── ingest.py              # PDF table extraction & batch indexing CLI
│
├── models/                    # Model Loaders & LLM Integrations
│   ├── __init__.py
│   ├── embeddings.py          # SentenceTransformer embedding service (optimized for 16GB RAM)
│   ├── image_classifier.py    # MobileNetV2 plant disease vision classifier
│   └── llm_client.py          # LLM inference client (Gemini / Hugging Face / OpenAI / Local fallback)
│
└── data/                      # Data Storage & Knowledge Base
    ├── __init__.py
    ├── sample_crop_guide.json # Default seed knowledge (crops, pests, schemes)
    ├── uploads/               # Uploaded documents directory
    └── chroma_db/             # Persistent ChromaDB vector storage
```


---

## 🛠️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-KI-SHAN.git
cd AI-KI-SHAN
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```
Edit `.env` with your preferred settings (e.g. `TELEGRAM_BOT_TOKEN`, `HUGGINGFACE_API_TOKEN`).

### 5. Run the server locally
```bash
python app.py
```
Or with Uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Interactive API documentation will be available at: `http://localhost:7860/docs`

---

## 🤗 Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Docker** or **Blank (Python)** / **FastAPI** SDK.
3. Push the codebase to your Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/AI-ki-shan
   git push space main
   ```
4. Configure Secret Environment Variables in your Hugging Face Space settings:
   - `TELEGRAM_BOT_TOKEN=your_telegram_bot_token`
   - `TELEGRAM_BOT_ENABLED=true`
   - `TELEGRAM_MODE=webhook`
   - `TELEGRAM_WEBHOOK_URL=https://your-username-ai-ki-shan.hf.space/webhook/telegram`
   - `GEMINI_API_KEY=your_gemini_api_key`
   - `HUGGINGFACE_API_TOKEN=your_hf_token`

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API service status and endpoint index |
| `GET` | `/health` | Health check & ChromaDB statistics |
| `GET` | `/api/stats` | Detailed vector store metrics & configuration |
| `POST` | `/api/chat` | Query the RAG engine for agriculture advice |
| `POST` | `/api/ingest` | Upload & index agricultural documents/tables |
| `POST` | `/webhook/telegram` | Telegram webhook receiver |

### Sample Chat Request
```bash
curl -X POST "http://localhost:7860/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "How to control yellow rust in wheat?", "crop": "Wheat"}'
```

---

## 🤖 Telegram Bot Commands

- `/start` - Launch the interactive main menu
- `/help` - View usage guide and sample queries
- `/schemes` - Browse government schemes (PM-KISAN, PMFBY)
- `/soil` - Soil testing & fertilizer management
- `/pests` - Pest identification & IPM advisory
