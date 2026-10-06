import os
import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from rag.pipeline import get_rag_pipeline
from rag.vector_store import get_vector_store
from bot.bot_service import get_bot_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_ki_shan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes vector database, seeds default knowledge base,
    and starts the Telegram bot if enabled.
    """
    logger.info("Initializing AI-ki-shan backend services...")

    # 1. Initialize RAG pipeline and seed initial agricultural knowledge
    try:
        rag = get_rag_pipeline()
        rag.seed_initial_knowledge()
    except Exception as e:
        logger.error(f"Error during RAG knowledge seeding: {e}", exc_info=True)

    # 2. Start Telegram Bot if polling is enabled
    bot_service = get_bot_service()
    if settings.TELEGRAM_BOT_ENABLED:
        if settings.TELEGRAM_MODE == "polling":
            asyncio.create_task(bot_service.start_polling())
        elif settings.TELEGRAM_MODE == "webhook":
            await bot_service.setup_webhook(settings.TELEGRAM_WEBHOOK_URL)

    yield

    # Teardown logic
    logger.info("Shutting down AI-ki-shan backend services...")
    if settings.TELEGRAM_BOT_ENABLED:
        await bot_service.stop()


app = FastAPI(
    title="AI-ki-shan (एआई-किसान) API",
    description="Agriculture AI Backend optimized for Hugging Face Spaces (16GB RAM environment).",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web UI / mobile integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Request & Response Schemas
# =====================================================================

class ChatRequest(BaseModel):
    query: str = Field(..., description="Agricultural query or farmer question", examples=["How to control yellow rust in wheat?"])
    category: Optional[str] = Field(None, description="Optional category filter (e.g., 'Pest & Disease Control', 'Government Scheme')")
    crop: Optional[str] = Field(None, description="Optional crop filter (e.g., 'Wheat', 'Rice', 'Cotton')")
    language: str = Field("en", description="Preferred response language ('en', 'hi')")


class SourceItem(BaseModel):
    title: str
    category: str
    crop: str
    score: Optional[float] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    retrieved_sources_count: int
    sources: List[SourceItem]


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service overview."""
    return {
        "service": "AI-ki-shan Agriculture AI Backend",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "chat": "POST /api/chat",
            "ingest": "POST /api/ingest",
            "health": "GET /health",
            "stats": "GET /api/stats"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check for monitoring and Hugging Face Spaces liveness probes."""
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "vector_store": stats,
        "telegram_bot_enabled": settings.TELEGRAM_BOT_ENABLED
    }


@app.get("/api/stats", tags=["Monitoring"])
async def get_stats():
    """Retrieves vector database and application metrics."""
    vector_store = get_vector_store()
    return {
        "vector_store": vector_store.get_stats(),
        "config": {
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "max_retrieved_docs": settings.MAX_RETRIEVED_DOCS,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "telegram_mode": settings.TELEGRAM_MODE
        }
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["RAG Chat"])
async def chat(request: ChatRequest):
    """
    Submits a query to the agricultural RAG pipeline and returns verified advice.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        rag = get_rag_pipeline()
        result = await rag.query(
            question=request.query,
            category=request.category,
            crop=request.crop,
            language=request.language
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Error handling chat query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process agricultural query: {str(e)}")


@app.post("/api/ingest", tags=["RAG Ingestion"])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    crop: Optional[str] = Form(None)
):
    """
    Ingests agricultural documents (PDF, CSV tables, TXT, JSON, MD) using 'unstructured'.
    """
    upload_dir = os.path.join(".", "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    safe_filename = os.path.basename(file.filename or "uploaded_doc")
    file_path = os.path.join(upload_dir, safe_filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        metadata = {}
        if category:
            metadata["category"] = category
        if crop:
            metadata["crop"] = crop

        rag = get_rag_pipeline()
        ingest_result = rag.ingest_file(file_path, extra_metadata=metadata)

        return {
            "message": f"File '{safe_filename}' processed and indexed into ChromaDB successfully.",
            "details": ingest_result
        }
    except Exception as e:
        logger.error(f"Error ingesting file {safe_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")



@app.post("/webhook/telegram", tags=["Telegram Webhook"])
async def telegram_webhook(request: Request):
    """
    Telegram Webhook receiver for production deployments on Hugging Face Spaces.
    """
    try:
        data = await request.json()
        bot_service = get_bot_service()
        await bot_service.process_webhook_update(data)
        return JSONResponse(content={"status": "ok"})
    except Exception as e:
        logger.error(f"Telegram webhook error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


if __name__ == "__main__":
    # Hugging Face Spaces binds to 0.0.0.0:7860 by default
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
