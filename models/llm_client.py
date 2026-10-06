import os
import logging
from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
import httpx
from config import settings

logger = logging.getLogger(__name__)

# System instructions
AGRICULTURE_SYSTEM_INSTRUCTION = (
    "You are 'AI-ki-shan' (एआई-किसान), an expert, empathetic, and knowledgeable agricultural AI assistant. "
    "Your mission is to provide accurate, practical, and safe advice to farmers, agronomists, and researchers. "
    "Always prioritize verified context. If you recommend chemicals or pesticides, include safety precautions "
    "and organic alternatives where available. Reply in the same language as the user's query."
)

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


class LLMClient:
    """
    LLM Client for AI-ki-shan agriculture chatbot.
    Supports Google Gemini, Hugging Face Inference API, OpenAI, and intelligent
    extractive fallbacks when external API keys are not supplied.
    """

    def __init__(self):
        self.gemini_api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.gemini_model = settings.GEMINI_MODEL
        self.hf_token = settings.HUGGINGFACE_API_TOKEN
        self.model_id = settings.HF_LLM_MODEL_ID
        self.openai_api_key = settings.OPENAI_API_KEY
        self._gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        """Initializes the Gemini client if API key is present."""
        if not self.gemini_api_key:
            return

        if GOOGLE_GENAI_AVAILABLE:
            try:
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info(f"Initialized google-genai Client with model '{self.gemini_model}'.")
            except Exception as e:
                logger.warning(f"Could not initialize google-genai client: {e}")
        elif LEGACY_GENAI_AVAILABLE:
            try:
                legacy_genai.configure(api_key=self.gemini_api_key)
                self._gemini_client = legacy_genai.GenerativeModel(
                    model_name=self.gemini_model,
                    system_instruction=AGRICULTURE_SYSTEM_INSTRUCTION
                )
                logger.info(f"Initialized legacy google.generativeai with model '{self.gemini_model}'.")
            except Exception as e:
                logger.warning(f"Could not initialize legacy google.generativeai: {e}")

    def _build_agriculture_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Constructs an agriculture-specific system and context prompt."""
        formatted_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "")
            source = chunk.get("metadata", {}).get("source", "Knowledge Base")
            page = chunk.get("metadata", {}).get("page_number", "-")
            formatted_context += f"\n[Source {i} - {source} (Page {page})]:\n{text}\n"

        prompt = (
            "You are 'AI-ki-shan' (एआई-किसान), an expert agricultural AI assistant.\n"
            "Use the verified context below to answer the user query accurately.\n"
            "If the context provides the answer, prioritize it. If you recommend chemicals or pesticides, "
            "always include safety precautions and organic alternatives where available.\n"
            "Respond in the same language the user queried in.\n\n"
            f"=== VERIFIED CONTEXT ===\n{formatted_context or 'No specific context retrieved.'}\n\n"
            f"=== USER QUERY ===\n{query}\n\n"
            "=== AI-KI-SHAN ADVICE ==="
        )
        return prompt

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: str = "en"
    ) -> str:
        """
        Generates an agricultural advisory response using the best available LLM provider.
        """
        prompt = self._build_agriculture_prompt(query, context_chunks)

        # 1. Try Google Gemini API if configured
        if self.gemini_api_key:
            try:
                response = await self._query_gemini(prompt)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Gemini API query failed: {e}. Trying next provider...")

        # 2. Try Hugging Face Inference API if token is configured
        if self.hf_token:
            try:
                response = await self._query_hf_api(prompt)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"HF Inference API query failed: {e}. Trying next provider...")

        # 3. Try OpenAI API if key is configured
        if self.openai_api_key:
            try:
                response = await self._query_openai(prompt)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"OpenAI query failed: {e}. Falling back to local synthesis...")

        # 4. Default: Fallback synthesis using retrieved context & domain heuristics
        return self._synthesize_local_response(query, context_chunks, language)

    async def _query_gemini(self, prompt: str) -> Optional[str]:
        """Query Gemini API via SDKs or direct REST fallback."""
        if not self.gemini_api_key:
            return None

        # 1. google-genai SDK
        if GOOGLE_GENAI_AVAILABLE and self._gemini_client:
            try:
                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=AGRICULTURE_SYSTEM_INSTRUCTION,
                        temperature=0.3,
                    )
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"google-genai SDK error in LLMClient: {e}")

        # 2. legacy google.generativeai SDK
        if LEGACY_GENAI_AVAILABLE and self._gemini_client:
            try:
                response = self._gemini_client.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"legacy google.generativeai error in LLMClient: {e}")

        # 3. Direct REST Fallback
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
            payload = {
                "system_instruction": {
                    "parts": [{"text": AGRICULTURE_SYSTEM_INSTRUCTION}]
                },
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3
                }
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
        except Exception as e:
            logger.warning(f"Direct Gemini REST API error in LLMClient: {e}")

        return None

    async def _query_hf_api(self, prompt: str) -> Optional[str]:
        """Query Hugging Face Inference API."""
        api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(api_url, json=payload, headers=headers)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "").strip()
                elif isinstance(data, dict):
                    return data.get("generated_text", "").strip()
            else:
                logger.warning(f"HF API returned status {res.status_code}: {res.text}")
        return None

    async def _query_openai(self, prompt: str) -> Optional[str]:
        """Query OpenAI Chat Completion API."""
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": AGRICULTURE_SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.3
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(api_url, json=payload, headers=headers)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
        return None

    def _synthesize_local_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: str = "en"
    ) -> str:
        """
        Extractive RAG response generator for standalone operation without API keys.
        """
        if not context_chunks:
            return (
                "🌾 **AI-ki-shan Agricultural Advisory** 🌾\n\n"
                f"I received your inquiry regarding: *{query}*.\n\n"
                "I could not locate direct documentation in the active knowledge base for this exact query. "
                "For detailed guidance, please specify the crop name, pest symptoms, soil type, or government scheme name.\n\n"
                "💡 *Tip: Ingest relevant crop guidelines and agricultural PDFs/tables via `/api/ingest`.*"
            )

        response_lines = ["🌾 **AI-ki-shan Agricultural Advisory** 🌾\n"]
        response_lines.append("Here is the recommended guidance based on verified agricultural data:\n")

        for idx, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})
            title = meta.get("title", f"Guideline {idx}")
            category = meta.get("category", "General Agriculture")
            score = chunk.get("score")
            
            score_text = ""
            if score is not None:
                # Clamp confidence between 0.0 and 1.0 (score is cosine distance)
                confidence = max(0.0, min(1.0, 1.0 - float(score)))
                score_text = f" (Confidence: {confidence:.1%})"

            response_lines.append(f"### 📋 {title} [{category}]{score_text}")
            response_lines.append(f"{text}\n")

        response_lines.append("---")
        response_lines.append("🌱 *Always verify chemical dosages with local Krishi Vigyan Kendra (KVK) or extension officers.*")

        return "\n".join(response_lines)


_llm_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Singleton getter for the LLMClient."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance
