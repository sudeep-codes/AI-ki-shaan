"""
Telegram Bot package for AI-ki-shan.
Provides interactive conversational interface for farmers and agriculture professionals.
"""
from bot.bot_service import TelegramBotService, get_bot_service
from bot.chatbot_pipeline import (
    TelegramChatbotPipeline,
    get_chatbot_pipeline,
    handle_telegram_chat,
    STRICT_SYSTEM_INSTRUCTION
)

__all__ = [
    "TelegramBotService",
    "get_bot_service",
    "TelegramChatbotPipeline",
    "get_chatbot_pipeline",
    "handle_telegram_chat",
    "STRICT_SYSTEM_INSTRUCTION"
]

