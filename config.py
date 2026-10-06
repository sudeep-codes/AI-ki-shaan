import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """
        Application Settings for AI-ki-shan backend.
        Optimized for 16GB RAM Hugging Face Spaces environment.
        """
        # Server configuration
        HOST: str = "0.0.0.0"
        PORT: int = 7860
        ENVIRONMENT: str = "production"
        DEBUG: bool = False
        LOG_LEVEL: str = "INFO"

        # Telegram Bot
        TELEGRAM_BOT_TOKEN: Optional[str] = None
        TELEGRAM_BOT_ENABLED: bool = False
        TELEGRAM_MODE: str = "polling"  # "polling" or "webhook"
        TELEGRAM_WEBHOOK_URL: Optional[str] = None

        # RAG & Embedding Settings
        EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
        CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
        CHROMA_COLLECTION_NAME: str = "agriculture_knowledge_base"
        CHROMA_FINANCE_COLLECTION_NAME: str = "finance_knowledge_base"
        MAX_RETRIEVED_DOCS: int = 4
        CONFIDENCE_THRESHOLD: float = 0.65

        # LLM Settings
        HUGGINGFACE_API_TOKEN: Optional[str] = None
        HF_LLM_MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"
        OPENAI_API_KEY: Optional[str] = None
        GEMINI_API_KEY: Optional[str] = None
        GEMINI_MODEL: str = "gemini-3.8-flash"

        # Agriculture domain settings
        DEFAULT_LANGUAGE: str = "en"

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )

    settings = Settings()

except ImportError:
    # Fallback when pydantic-settings is not yet installed
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    class FallbackSettings:
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", "7860"))
        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
        DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")
        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        TELEGRAM_BOT_ENABLED: bool = os.getenv("TELEGRAM_BOT_ENABLED", "false").lower() in ("true", "1")
        TELEGRAM_MODE: str = os.getenv("TELEGRAM_MODE", "polling")
        TELEGRAM_WEBHOOK_URL: Optional[str] = os.getenv("TELEGRAM_WEBHOOK_URL")

        EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "agriculture_knowledge_base")
        CHROMA_FINANCE_COLLECTION_NAME: str = os.getenv("CHROMA_FINANCE_COLLECTION_NAME", "finance_knowledge_base")
        MAX_RETRIEVED_DOCS: int = int(os.getenv("MAX_RETRIEVED_DOCS", "4"))
        CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

        HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
        HF_LLM_MODEL_ID: str = os.getenv("HF_LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
        GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

        DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")

    settings = FallbackSettings()
