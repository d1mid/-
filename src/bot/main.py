from src.bot.core.bot import PlumbingBot


def main() -> None:
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
