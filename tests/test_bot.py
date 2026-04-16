from src.bot.core.bot import PlumbingBot


def test_bot_handles_catalog_request() -> None:
    bot = PlumbingBot()
    result = bot.reply("покажи каталог")
    assert result["intent"] == "show_catalog"
    assert "категории" in result["answer"].lower() or "каталоге" in result["answer"].lower()


def test_bot_handles_budget_selection() -> None:
    bot = PlumbingBot()
    result = bot.reply("что есть до 5000 рублей")
    assert result["intent"] == "selection_by_budget"
    assert "5000" in result["answer"] or "руб" in result["answer"].lower()


def test_bot_handles_product_price() -> None:
    bot = PlumbingBot()
    result = bot.reply("сколько стоит AquaMix ProFilter K-500")
    assert result["intent"] == "ask_price"
    assert "9490" in result["answer"] or "руб" in result["answer"].lower()


def test_bot_handles_product_characteristics() -> None:
    bot = PlumbingBot()
    result = bot.reply("расскажи подробнее про AquaMix RainTherm S-900")
    assert result["intent"] in {"ask_characteristics", "select_shower_system"}
    assert "характерист" in result["answer"].lower() or "термостат" in result["answer"].lower()


def test_bot_uses_dialogue_answer_for_non_catalog_phrase() -> None:
    bot = PlumbingBot()
    result = bot.reply("ты бот")
    assert result["answer"]


def test_bot_handles_identity_intent() -> None:
    bot = PlumbingBot()
    result = bot.reply("ты бот")
    assert result["intent"] == "ask_bot_identity"
    assert "бот" in result["answer"].lower() or "консультант" in result["answer"].lower()


def test_bot_handles_capabilities_intent() -> None:
    bot = PlumbingBot()
    result = bot.reply("что ты умеешь")
    assert result["intent"] == "bot_capabilities"
    assert "могу" in result["answer"].lower() or "помога" in result["answer"].lower()


def test_bot_handles_complete_set_intent() -> None:
    bot = PlumbingBot()
    result = bot.reply("нужен комплект для ванной")
    assert result["intent"] == "select_complete_set"
    assert "комплект" in result["answer"].lower()


def test_bot_handles_small_talk() -> None:
    bot = PlumbingBot()
    result = bot.reply("Как у тебя дела?")
    assert result["intent"] == "small_talk"
    assert "хорош" in result["answer"].lower() or "спасибо" in result["answer"].lower()


def test_bot_can_softly_transition_to_promo_after_small_talk() -> None:
    bot = PlumbingBot()
    bot.reply("Как у тебя дела?", conversation_id="chat-1")
    result = bot.reply("Что нового?", conversation_id="chat-1")
    assert result["intent"] == "small_talk"
    assert "кстати" in result["answer"].lower() or "каталог" in result["answer"].lower()


def test_bot_adds_soft_promo_to_recommendation() -> None:
    bot = PlumbingBot()
    result = bot.reply("что посоветуешь", conversation_id="chat-2")
    assert result["intent"] == "request_recommendation"
    assert "кстати" in result["answer"].lower() or "руб." in result["answer"].lower()


def test_bot_handles_category_request_for_installation() -> None:
    bot = PlumbingBot()
    result = bot.reply("что есть из систем монтажа")
    assert "Системы монтажа" in result["answer"] or "SanFrame" in result["answer"]


def test_bot_handles_category_request_for_water_heaters() -> None:
    bot = PlumbingBot()
    result = bot.reply("покажи водонагреватели")
    assert "HeatLine" in result["answer"] or "Водонагреватели" in result["answer"]
