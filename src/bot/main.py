"""Точка входа для локального запуска CLI-бота и Telegram-режима."""

from __future__ import annotations

import sys

from src.bot.core.bot import PlumbingBot
from src.bot.telegram.handler import run_telegram_bot


def main() -> None:
    # CLI-режим оставлен как самый простой способ локальной проверки без Telegram.
    if len(sys.argv) > 1 and sys.argv[1] == "--telegram":
        run_telegram_bot()
        return

    bot = PlumbingBot()
    print("Бот по сантехнике запущен. Для выхода введите пустую строку.")

    while True:
        replica = input("> ").strip()
        if not replica:
            break
        result = bot.reply(replica)
        print(f"< {result['answer']}")


if __name__ == "__main__":
    main()
