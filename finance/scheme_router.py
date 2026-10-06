import os
import re
import logging
from typing import Optional

try:
    from config import settings
    DEFAULT_API_KEY = settings.GEMINI_API_KEY
    DEFAULT_MODEL = settings.GEMINI_MODEL
except Exception:
    DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY")
    DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

logger = logging.getLogger(__name__)

# Keywords for rule-based fallback intent classification
SCHEME_KEYWORDS = [
    r"\b(?:pm[- ]?kisan|pmkisan|kisan samman nidhi)\b",
    r"\b(?:pmfby|fasal bima|crop insurance|bima)\b",
    r"\b(?:kcc|kisan credit card|credit card)\b",
    r"\b(?:loan|loans|karz|rin|kisan loan)\b",
    r"\b(?:subsidy|subsidies|anudan|sahayata)\b",
    r"\b(?:yojana|scheme|schemes|sarkari yojana)\b",
    r"\b(?:grant|financial|pension|dbt|installment|kist|paisa|money)\b",
]

AGRI_KEYWORDS = [
    r"\b(?:wheat|rice|paddy|cotton|mustard|gehun|dhan|kapas|sarson|crop|crops|fasal)\b",
    r"\b(?:fertilizer|fertilizers|urea|dap|npk|potash|khad|manure|fym|vermicompost)\b",
    r"\b(?:pest|pests|insect|insects|disease|blight|rust|aphid|bollworm|keeda|bimari|fungus)\b",
    r"\b(?:soil|soil test|ph|saline|organic carbon|mitti|irrigation|sinchai|seed|beej)\b",
    r"\b(?:spray|dosage|pesticide|fungicide|insecticide|herbicide|dawai)\b",
]

_model_instance = None


def _get_gemini_model():
    """Lazily initializes the Gemini GenerativeModel if configured."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    api_key = DEFAULT_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _model_instance = genai.GenerativeModel(DEFAULT_MODEL)
        return _model_instance
    except Exception as e:
        logger.warning(f"Could not initialize Gemini model in scheme_router: {e}")
        return None


def rule_based_intent(user_message: str) -> str:
    """Fast, accurate rule-based classification fallback."""
    msg = user_message.lower()

    scheme_score = sum(len(re.findall(pattern, msg)) for pattern in SCHEME_KEYWORDS)
    agri_score = sum(len(re.findall(pattern, msg)) for pattern in AGRI_KEYWORDS)

    # If explicit government scheme, financial, loan, or insurance terms appear, route to SCHEME
    if scheme_score > 0:
        return "SCHEME"
    return "AGRICULTURE"



def detect_intent(user_message: str) -> str:
    """
    Classifies a user query into 'SCHEME' or 'AGRICULTURE'.
    Uses Gemini API if available, with robust heuristic fallback.
    """
    if not user_message or not user_message.strip():
        return "AGRICULTURE"

    model = _get_gemini_model()
    if model:
        prompt = f"""
Analyze this user message: "{user_message}"
Classify it into exactly one of these two categories:
1. 'SCHEME' (if it mentions government schemes, loans, KCC, PM-KISAN, crop insurance, PMFBY, money, subsidies, financial benefits)
2. 'AGRICULTURE' (if it mentions crops, fertilizer, pests, farming techniques, disease, soil health, irrigation)

Return ONLY the category word: SCHEME or AGRICULTURE.
"""
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                category = response.text.strip().upper()
                if "SCHEME" in category:
                    return "SCHEME"
                elif "AGRICULTURE" in category:
                    return "AGRICULTURE"
        except Exception as e:
            logger.warning(f"Gemini intent classification failed: {e}. Using rule-based fallback.")

    return rule_based_intent(user_message)