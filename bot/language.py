import os
import json
import logging
import httpx
from typing import Dict

try:
    from config import settings
    DEFAULT_API_KEY = settings.GEMINI_API_KEY
    DEFAULT_MODEL = settings.GEMINI_MODEL
except Exception:
    DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY")
    DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

logger = logging.getLogger(__name__)

async def detect_and_translate(user_message: str) -> Dict[str, str]:
    """
    Detects the language of the user message.
    If it's not English, translates it to English.
    Returns a dictionary with 'language' (the detected language) and 'english_text'.
    """
    if not user_message or not user_message.strip():
        return {"language": "English", "english_text": ""}

    api_key = DEFAULT_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found. Skipping translation.")
        return {"language": "Unknown", "english_text": user_message}

    prompt = f"""
Analyze the following text: "{user_message}"

1. Identify the language of the text.
2. If the text is in English, return the exact same text.
3. If the text is NOT in English, translate it to English.

Return a valid JSON object strictly matching this schema:
{{
    "language": "Detected language (e.g., English, Hindi, Marathi, etc.)",
    "english_text": "The translated text, or the original text if it was already English"
}}
"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json"
            }
        }
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            res = await http_client.post(url, json=payload)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        response_text = parts[0].get("text", "").strip()
                        result = json.loads(response_text)
                        return {
                            "language": result.get("language", "English"),
                            "english_text": result.get("english_text", user_message)
                        }
            else:
                logger.error(f"Translation API error {res.status_code}: {res.text}")
    except Exception as e:
        logger.error(f"Error in detect_and_translate: {e}", exc_info=True)
    
    # Fallback in case of failure
    return {"language": "Unknown", "english_text": user_message}
