import asyncio
import logging
from typing import Optional

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        filters
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from config import settings
from bot.handlers import (
    start_command,
    help_command,
    language_command,
    finance_command,
    schemes_command,
    soil_command,
    pests_command,
    button_callback_handler,
    text_message_handler,
    handle_photo
)

logger = logging.getLogger(__name__)


class TelegramBotService:
    """
    Manages the lifecycle and routing of the Telegram Bot for AI-ki-shan.
    Supports asynchronous polling or FastAPI webhook processing.
    """

    def __init__(self):
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.enabled = settings.TELEGRAM_BOT_ENABLED and bool(self.token)
        self.mode = settings.TELEGRAM_MODE
        self.application: Optional[Application] = None
        self._is_running = False

    def setup(self) -> bool:
        """Initializes the Telegram Application and registers all handlers."""
        if not self.enabled:
            logger.info("Telegram bot is disabled or TELEGRAM_BOT_TOKEN is not set.")
            return False

        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot library is not installed.")
            return False

        try:
            self.application = ApplicationBuilder().token(self.token).build()

            # Register Command Handlers
            self.application.add_handler(CommandHandler("start", start_command))
            self.application.add_handler(CommandHandler("help", help_command))
            self.application.add_handler(CommandHandler("language", language_command))
            self.application.add_handler(CommandHandler("finance", finance_command))
            self.application.add_handler(CommandHandler("schemes", schemes_command))
            self.application.add_handler(CommandHandler("soil", soil_command))
            self.application.add_handler(CommandHandler("pests", pests_command))

            # Register Callback Query Handlers (Inline Buttons)
            self.application.add_handler(CallbackQueryHandler(button_callback_handler))

            # Register Photo Handler (Crop disease diagnosis)
            self.application.add_handler(
                MessageHandler(filters.PHOTO, handle_photo)
            )

            # Register Text Message Handler (RAG questions)
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler)
            )

            logger.info("Telegram Bot handlers successfully registered.")
            return True

        except Exception as e:
            logger.error(f"Failed to setup Telegram Bot: {e}", exc_info=True)
            return False

    async def start_polling(self):
        """Starts the bot in background polling mode."""
        if not self.application:
            if not self.setup():
                return

        try:
            logger.info("Starting Telegram Bot in Polling Mode...")
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(drop_pending_updates=True)
            self._is_running = True
            logger.info("Telegram Bot polling started successfully.")
        except Exception as e:
            logger.error(f"Error starting Telegram Bot polling: {e}", exc_info=True)

    async def setup_webhook(self, webhook_url: str):
        """Initializes the bot and sets the Telegram webhook."""
        if not webhook_url:
            logger.error("TELEGRAM_WEBHOOK_URL is required for webhook mode.")
            return

        if not self.application:
            if not self.setup():
                return

        try:
            logger.info(f"Setting up Telegram Webhook to {webhook_url}...")
            await self.application.initialize()
            await self.application.start()
            await self.application.bot.set_webhook(url=webhook_url)
            self._is_running = True
            logger.info("Telegram Webhook set successfully.")
        except Exception as e:
            logger.error(f"Error setting Telegram Webhook: {e}", exc_info=True)

    async def stop(self):
        """Gracefully stops the Telegram bot."""
        if self.application and self._is_running:
            logger.info("Stopping Telegram Bot...")
            try:
                if self.application.updater and self.application.updater.running:
                    await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                self._is_running = False
                logger.info("Telegram Bot stopped.")
            except Exception as e:
                logger.error(f"Error stopping Telegram Bot: {e}")

    async def process_webhook_update(self, update_data: dict):
        """Processes an incoming Telegram webhook update."""
        if not self.application:
            if not self.setup():
                return
        try:
            update = Update.de_json(data=update_data, bot=self.application.bot)
            await self.application.process_update(update)
        except Exception as e:
            logger.error(f"Error processing webhook update: {e}", exc_info=True)


_bot_service_instance: Optional[TelegramBotService] = None


def get_bot_service() -> TelegramBotService:
    """Singleton getter for the TelegramBotService."""
    global _bot_service_instance
    if _bot_service_instance is None:
        _bot_service_instance = TelegramBotService()
    return _bot_service_instance
