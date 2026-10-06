import logging
from typing import Optional, Any

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ContextTypes
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = Any
    InlineKeyboardButton = Any
    InlineKeyboardMarkup = Any
    class MockContextTypes:
        DEFAULT_TYPE = Any
    ContextTypes = MockContextTypes
    class ParseMode:
        MARKDOWN = "Markdown"



from rag.pipeline import get_rag_pipeline
from bot.chatbot_pipeline import get_chatbot_pipeline
from bot.telegram_handlers import handle_photo, handle_text

logger = logging.getLogger(__name__)



async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /start command with a welcome message and interactive keyboard."""
    user = update.effective_user
    name = user.first_name if user else "Kisan Mitra"

    welcome_text = (
        f"🙏 **Namaste {name}! Welcome to AI-ki-shan (एआई-किसान)** 🌾\n\n"
        "I am your dedicated AI Agriculture Assistant, here to assist with:\n"
        "🌱 **Crop Health & Disease Diagnosis**\n"
        "🧪 **Fertilizer & Soil Management**\n"
        "🐛 **Pest Control & Organic Remedies**\n"
        "📜 **Government Schemes** (PM-KISAN, PMFBY, KCC)\n"
        "💧 **Irrigation & Water Conservation**\n\n"
        "Ask any question in any language (English, हिंदी, मराठी, తెలుగు, ਪੰਜਾਬੀ, etc.), or choose a topic below:"
    )

    keyboard = [
        [
            InlineKeyboardButton("🌾 Crop Disease & Pests", callback_data="topic_pests"),
            InlineKeyboardButton("🧪 Soil & Fertilizers", callback_data="topic_soil")
        ],
        [
            InlineKeyboardButton("📜 Government Schemes", callback_data="topic_schemes"),
            InlineKeyboardButton("💡 How to Use", callback_data="topic_help")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        text=welcome_text,
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /language command."""
    lang_text = (
        "🌐 **Multilingual Support Enabled** 🌐\n\n"
        "AI-ki-shan automatically detects your language!\n"
        "You can chat in **English, Hindi, Marathi, Telugu, Punjabi, etc.**\n"
        "Just type your query naturally and the bot will reply in the same language."
    )
    await update.message.reply_text(text=lang_text, parse_mode=ParseMode.MARKDOWN)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /help command."""
    help_text = (
        "🌾 **AI-ki-shan Command Guide** 🌾\n\n"
        "• `/start` - Restart the bot & display main menu\n"
        "• `/help` - View this help guide\n"
        "• `/language` - View supported languages\n"
        "• `/finance` - Explore PM-KISAN, PMFBY & agriculture schemes\n"
        "• `/schemes` - Alias for /finance\n"
        "• `/soil` - Soil health & NPK recommendations\n"
        "• `/pests` - Pest identification & IPM advice\n\n"
        "💬 **Direct Inquiries:** Simply send any farming query in English, Hindi, or any native Indian language!\n"
        "Example: *'How to control yellow rust in wheat?'* or *'पीएम-किसान योजना के क्या लाभ हैं?'*"
    )
    await update.message.reply_text(text=help_text, parse_mode=ParseMode.MARKDOWN)


async def schemes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /schemes command."""
    await finance_command(update, context)

async def finance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /finance command."""
    chatbot = get_chatbot_pipeline()
    result = await chatbot.process_telegram_message("What are the key government agriculture schemes PM-KISAN and PMFBY?", intent="SCHEME")
    await update.message.reply_text(text=result["answer"], parse_mode=ParseMode.MARKDOWN)


async def soil_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /soil command."""
    chatbot = get_chatbot_pipeline()
    result = await chatbot.process_telegram_message("How to manage soil health and fertilizer balance?")
    await update.message.reply_text(text=result["answer"], parse_mode=ParseMode.MARKDOWN)


async def pests_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /pests command."""
    text = (
        "🐛 **Pest & Disease Advisory**\n\n"
        "Please specify your crop and visible symptoms:\n"
        "• E.g.: *'Mustard aphid control'* or *'Rice blast treatment'* or *'Cotton pink bollworm'*."
    )
    await update.message.reply_text(text=text, parse_mode=ParseMode.MARKDOWN)


async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles inline button callbacks."""
    query = update.callback_query
    await query.answer()

    data = query.data
    chatbot = get_chatbot_pipeline()

    if data == "topic_pests":
        await query.message.reply_text(
            "🐛 **Pest Management**: Please type your crop name and symptoms (e.g., 'Wheat yellow rust' or 'Cotton pink bollworm')."
        )
    elif data == "topic_soil":
        result = await chatbot.process_telegram_message("Soil health testing, FYM, and NPK fertilizer guidelines")
        await query.message.reply_text(text=result["answer"], parse_mode=ParseMode.MARKDOWN)
    elif data == "topic_schemes":
        result = await chatbot.process_telegram_message("PM-KISAN and PMFBY government schemes details", intent="SCHEME")
        await query.message.reply_text(text=result["answer"], parse_mode=ParseMode.MARKDOWN)
    elif data == "topic_help":
        await help_command(update, context)


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles freeform user text inquiries via TelegramChatbotPipeline:
    1. Queries ChromaDB for context (tables & text)
    2. Constructs prompt with retrieved context
    3. Queries Gemini API with strict system instructions and native translation
    """
    if not update.message or not update.message.text:
        return

    user_query = update.message.text
    logger.info(f"Telegram Query from User {update.effective_user.id}: {user_query}")

    # Send typing indicator
    await update.message.chat.send_action(action="typing")

    try:
        from finance.scheme_router import detect_intent
        intent = detect_intent(user_query)

        chatbot = get_chatbot_pipeline()
        result = await chatbot.process_telegram_message(message_text=user_query, intent=intent)
        answer = result.get("answer", "I do not know.")

        # Ensure message stays within Telegram's 4096 character limit
        if len(answer) > 4000:
            answer = answer[:3950] + "\n\n...(truncated)"

        await update.message.reply_text(text=answer, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error handling Telegram message: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ An error occurred while retrieving agricultural advice. Please try again in a moment."
        )

