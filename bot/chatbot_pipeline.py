import logging
from typing import List, Dict, Any, Optional

from config import settings
from rag.ingest import query_top_k

logger = logging.getLogger("bot.chatbot_pipeline")

# Strict System Instruction as specified
STRICT_SYSTEM_INSTRUCTION = (
    "You are an agricultural assistant. "
    "Answer ONLY based on the provided context. "
    "If the context lacks the answer, state that you do not know. "
    "Reply in the exact same language the user used."
)

# Setup Gemini SDK clients
try:
    from google import genai
    from google.genai import types
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

try:
    import google.generativeai as legacy_genai
    LEGACY_GENAI_AVAILABLE = True
except ImportError:
    legacy_genai = None
    LEGACY_GENAI_AVAILABLE = False


class TelegramChatbotPipeline:
    """
    Agricultural Chatbot Pipeline for Telegram messages:
    1. Queries ChromaDB from /rag to retrieve relevant agricultural context (tables + text).
    2. Builds a structured prompt incorporating verified context chunks.
    3. Invokes Gemini API with strict system instructions and native multilingual response.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini Client."""
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini API calls will require an API key in .env.")
            return

        if GOOGLE_GENAI_AVAILABLE:
            try:
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Google GenAI client with model '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize google-genai client: {e}")
        elif LEGACY_GENAI_AVAILABLE:
            try:
                legacy_genai.configure(api_key=self.api_key)
                self._client = legacy_genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=STRICT_SYSTEM_INSTRUCTION
                )
                logger.info(f"Initialized legacy google.generativeai client with model '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to initialize legacy google.generativeai client: {e}")

    def build_context_prompt(self, user_message: str, context_chunks: List[Dict[str, Any]], target_language: str = "English") -> str:
        """
        Constructs the context block containing retrieved text & Markdown tables,
        followed by the user's inquiry.
        """
        if not context_chunks:
            context_section = "No relevant context found in the database."
        else:
            context_items = []
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk.get("source", "Knowledge Base")
                page = chunk.get("page_number", "-")
                chunk_type = "Table" if chunk.get("is_table") else "Text"
                text = chunk.get("text", "").strip()

                header = f"[Source {i}: {source} | Page: {page} | Type: {chunk_type}]"
                context_items.append(f"{header}\n{text}")

            context_section = "\n\n---\n\n".join(context_items)

        prompt = (
            f"=== PROVIDED AGRICULTURAL CONTEXT ===\n"
            f"{context_section}\n\n"
            f"=== USER QUERY ===\n"
            f"{user_message}\n\n"
            f"IMPORTANT: You MUST respond in {target_language} language."
        )
        return prompt

    async def call_gemini(self, prompt: str) -> str:
        """Calls Gemini API using google-genai or fallback methods."""
        if not self.api_key:
            return (
                "⚠️ **Gemini API Key Missing**\n\n"
                "Please configure `GEMINI_API_KEY` in your `.env` file to enable AI answers."
            )

        # 1. Try modern google-genai SDK
        if GOOGLE_GENAI_AVAILABLE and self._client:
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=STRICT_SYSTEM_INSTRUCTION,
                        temperature=0.2,
                    )
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Error calling google-genai SDK: {e}", exc_info=True)

        # 2. Try legacy google.generativeai SDK
        if LEGACY_GENAI_AVAILABLE and self._client:
            try:
                response = self._client.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Error calling legacy google.generativeai SDK: {e}", exc_info=True)

        # 3. Direct REST Fallback via httpx
        try:
            import httpx
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "system_instruction": {
                    "parts": [{"text": STRICT_SYSTEM_INSTRUCTION}]
                },
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2
                }
            }
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                res = await http_client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                else:
                    logger.error(f"Gemini REST API error {res.status_code}: {res.text}")
        except Exception as e:
            logger.error(f"Gemini REST API fallback error: {e}", exc_info=True)

        return "I do not know."

    async def process_telegram_message(self, message_text: str, top_k: int = 5, intent: str = "AGRICULTURE") -> Dict[str, Any]:
        """
        Executes the full pipeline for an incoming Telegram message:
        1. Retrieve top-k context chunks from ChromaDB.
        2. Construct context-enriched prompt.
        3. Call Gemini API with strict agricultural system instructions.
        4. Return response and metadata.
        """
        if not message_text or not message_text.strip():
            return {
                "answer": "Please provide a valid question.",
                "context_chunks": [],
                "sources": []
            }

        logger.info(f"Processing Telegram message: '{message_text}'")

        # Step 0: Detect Language & Translate
        from bot.language import detect_and_translate
        lang_info = await detect_and_translate(message_text)
        detected_language = lang_info.get("language", "English")
        english_query = lang_info.get("english_text", message_text)

        logger.info(f"Language: {detected_language} | Translated Query: {english_query}")

        # Step 1: Choose collection based on intent
        collection_name = settings.CHROMA_COLLECTION_NAME
        if intent == "SCHEME":
            collection_name = settings.CHROMA_FINANCE_COLLECTION_NAME

        # Step 2: Query ChromaDB from /rag
        context_chunks = query_top_k(
            query_text=english_query,
            top_k=top_k,
            chroma_dir=settings.CHROMA_PERSIST_DIRECTORY,
            collection_name=collection_name,
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        logger.info(f"Retrieved {len(context_chunks)} context chunks from ChromaDB ({collection_name}).")

        # Step 3: Construct prompt with retrieved context
        prompt = self.build_context_prompt(
            user_message=english_query, 
            context_chunks=context_chunks, 
            target_language=detected_language
        )

        # Step 4: Call Gemini API (relying on native multilingual capabilities)
        answer = await self.call_gemini(prompt)

        # Step 5: Extract source summaries
        sources = [
            {
                "source": chunk.get("source"),
                "page": chunk.get("page_number"),
                "is_table": chunk.get("is_table", False),
                "similarity_score": chunk.get("similarity_score")
            }
            for chunk in context_chunks
        ]

        return {
            "query": message_text,
            "answer": answer,
            "context_chunks": context_chunks,
            "sources": sources
        }

    async def process_image_remedy(self, disease_name: str, confidence: float, target_language: str = "English") -> str:
        """
        Retrieves context for a detected plant disease and generates a remedy in the target language,
        using the specific prompt template from design.md.
        """
        logger.info(f"Processing image remedy for '{disease_name}' in {target_language}")

        query_text = f"{disease_name} symptoms and treatment remedies"
        context_chunks = query_top_k(
            query_text=query_text,
            top_k=5,
            chroma_dir=settings.CHROMA_PERSIST_DIRECTORY,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            model_name=settings.EMBEDDING_MODEL_NAME
        )

        if not context_chunks:
            context_section = "No relevant context found in the database."
        else:
            context_items = []
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk.get("source", "Knowledge Base")
                page = chunk.get("page_number", "-")
                text = chunk.get("text", "").strip()
                context_items.append(f"[Source {i}: {source} | Page: {page}]\n{text}")
            context_section = "\n\n---\n\n".join(context_items)

        prompt = (
            f"System: A plant disease classifier identified \"{disease_name}\" "
            f"with {confidence:.1%} confidence. Using the CONTEXT below, explain "
            f"what this is in simple terms and suggest a remedy. If the context "
            f"doesn't cover this specific issue, say so. Respond entirely in {target_language}.\n"
            f"Format your response with clear headings (e.g., '🌱 Diagnosis & Remedy Report 🌱', 'Condition', 'Recommended Treatment').\n\n"
            f"CONTEXT:\n{context_section}"
        )

        answer = await self.call_gemini(prompt)
        return answer



_chatbot_pipeline_instance: Optional[TelegramChatbotPipeline] = None


def get_chatbot_pipeline() -> TelegramChatbotPipeline:
    """Singleton getter for TelegramChatbotPipeline."""
    global _chatbot_pipeline_instance
    if _chatbot_pipeline_instance is None:
        _chatbot_pipeline_instance = TelegramChatbotPipeline()
    return _chatbot_pipeline_instance


async def handle_telegram_chat(message_text: str, intent: str = "AGRICULTURE") -> str:
    """Helper function to process a Telegram message string and return the answer."""
    pipeline = get_chatbot_pipeline()
    result = await pipeline.process_telegram_message(message_text, intent=intent)
    return result.get("answer", "I do not know.")
