"""Telegram-обертка над общей функцией бота."""

from __future__ import annotations

import os

from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from src.bot.core.bot import PlumbingBot


# Создает и настраивает Telegram-приложение с обработчиками команд и сообщений.
def build_telegram_application(token: str, bot: PlumbingBot | None = None) -> Application:
    plumbing_bot = bot or PlumbingBot()
    application = ApplicationBuilder().token(token).build()
    application.bot_data["plumbing_bot"] = plumbing_bot

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("catalog", show_catalog))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return application


# Отправляет стартовое приветствие при команде /start.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Здравствуйте! Я бот по продаже сантехники. Могу помочь с каталогом, подбором, ценами и характеристиками."
    )


# Отправляет краткую справку по возможностям бота при команде /help.
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Напишите, что вам нужно: смеситель, душевая система, раковина, унитаз, инсталляция или водонагреватель. "
        "Также можно спросить про цену, характеристики, бюджет или рекомендации."
    )


# Показывает каталог товаров при команде /catalog.
async def show_catalog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot = context.application.bot_data["plumbing_bot"]
    conversation_id = str(update.effective_chat.id) if update.effective_chat else "default"
    result = bot.reply("покажи каталог", conversation_id=conversation_id)
    await update.message.reply_text(result["answer"])


# Обрабатывает обычные текстовые сообщения пользователя через общую функцию бота.
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or not update.message.text:
        return

    # Chat ID используем как ключ сессии, чтобы сохранялась локальная память диалога.
    bot = context.application.bot_data["plumbing_bot"]
    conversation_id = str(update.effective_chat.id) if update.effective_chat else "default"
    result = bot.reply(update.message.text, conversation_id=conversation_id)
    await update.message.reply_text(result["answer"])


# Запускает Telegram-бота в режиме polling.
def run_telegram_bot(token: str | None = None) -> None:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Не найден токен Telegram. Укажите его через аргумент функции или переменную окружения TELEGRAM_BOT_TOKEN."
        )

    application = build_telegram_application(token)
    application.run_polling()
