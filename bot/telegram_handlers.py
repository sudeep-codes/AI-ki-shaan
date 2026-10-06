import os
import uuid
import logging
from typing import Any


try:
    from telegram import Update
    from telegram.ext import ContextTypes
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = Any
    class MockContextTypes:
        DEFAULT_TYPE = Any
    ContextTypes = MockContextTypes
    class ParseMode:
        MARKDOWN = "Markdown"



from models.image_classifier import diagnose_plant_image
from finance.scheme_router import detect_intent
from bot.chatbot_pipeline import get_chatbot_pipeline
from bot.language import detect_and_translate

logger = logging.getLogger(__name__)


# --- 1. PHOTO HANDLER ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles leaf & crop photo uploads from farmers:
    1. Downloads image securely to temporary storage.
    2. Runs plant disease identification via MobileNetV2.
    3. If disease identified, queries RAG pipeline for verified treatment remedies.
    4. Cleans up temporary image files safely.
    """
    if not update.message or not update.message.photo:
        return

    # Send typing / upload_photo indicator
    await update.message.chat.send_action(action="upload_photo")

    user_id = update.effective_user.id if update.effective_user else "anonymous"
    temp_filename = f"temp_crop_{user_id}_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join(".", "data", temp_filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    try:
        # Detect language from caption if provided
        target_language = "English"
        if update.message.caption:
            lang, _ = detect_and_translate(update.message.caption)
            target_language = lang

        # Telegram sends multiple sizes; [-1] gets the highest resolution
        photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
        await photo_file.download_to_drive(custom_path=temp_path)

        # Run plant disease classification
        diagnosis = diagnose_plant_image(temp_path)

        if not diagnosis.get("success"):
            msg = diagnosis.get(
                "message",
                "Could not clearly identify the crop issue. Please provide a clear, close-up photo of the affected leaf."
            )
            
            # Translate failure message if not English
            if target_language.lower() != "english":
                chatbot = get_chatbot_pipeline()
                msg = await chatbot.call_gemini(f"Translate this to {target_language}:\n{msg}")

            await update.message.reply_text(f"🔍\n\n{msg}", parse_mode=ParseMode.MARKDOWN)
            return

        disease_name = diagnosis["disease"]
        confidence = diagnosis["confidence"]

        # Inform farmer of diagnosis
        status_msg = (
            f"🔍 **Crop Health Diagnosis** 🌾\n\n"
            f"• **Detected Condition:** `{disease_name}`\n"
            f"• **Model Confidence:** `{confidence:.1%}`\n\n"
            f"⏳ *Retrieving verified agricultural treatment & management guide...*"
        )
        if target_language.lower() != "english":
            chatbot = get_chatbot_pipeline()
            status_msg = await chatbot.call_gemini(f"Translate this to {target_language}:\n{status_msg}")

        await update.message.reply_text(status_msg, parse_mode=ParseMode.MARKDOWN)

        # Query RAG knowledge base for verified treatment & remedies
        chatbot = get_chatbot_pipeline()
        final_response = await chatbot.process_image_remedy(disease_name, confidence, target_language)

        if len(final_response) > 4000:
            final_response = final_response[:3950] + "\n\n...(truncated)"

        await update.message.reply_text(final_response, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error during photo diagnosis handler: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ An error occurred while diagnosing the crop image. Please try again with a clear photo."
        )
    finally:
        # Clean up file immediately to avoid storage leaks
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file '{temp_path}': {e}")


# --- 2. TEXT HANDLER ---
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles user text messages with intent routing and RAG query execution.
    """
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    await update.message.chat.send_action(action="typing")

    try:
        # Detect if they are asking about government schemes or crop agriculture
        intent = detect_intent(user_text)
        logger.info(f"Detected intent '{intent}' for message: '{user_text[:50]}...'")

        chatbot = get_chatbot_pipeline()
        result = await chatbot.process_telegram_message(user_text, intent=intent)
        answer = result.get("answer", "I do not know.")

        if len(answer) > 4000:
            answer = answer[:3950] + "\n\n...(truncated)"

        await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error processing Telegram text message: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ An error occurred while processing your question. Please try again."
        )