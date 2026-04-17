"""Проверки общей функции бота и основных пользовательских сценариев."""

from src.bot.core.bot import PlumbingBot


# Проверяет, что бот корректно показывает каталог.
def test_bot_handles_catalog_request() -> None:
    bot = PlumbingBot()
    result = bot.reply("покажи каталог")
    assert result["intent"] == "show_catalog"
    assert "категории" in result["answer"].lower() or "каталоге" in result["answer"].lower()


# Проверяет подбор товаров по бюджету.
def test_bot_handles_budget_selection() -> None:
    bot = PlumbingBot()
    result = bot.reply("что есть до 5000 рублей")
    assert result["intent"] == "selection_by_budget"
    assert "5000" in result["answer"] or "руб" in result["answer"].lower()


# Проверяет ответ на вопрос о цене конкретной модели.
def test_bot_handles_product_price() -> None:
    bot = PlumbingBot()
    result = bot.reply("сколько стоит AquaMix ProFilter K-500")
    assert result["intent"] == "ask_price"
    assert "9490" in result["answer"] or "руб" in result["answer"].lower()


# Проверяет ответ с характеристиками товара.
def test_bot_handles_product_characteristics() -> None:
    bot = PlumbingBot()
    result = bot.reply("расскажи подробнее про AquaMix RainTherm S-900")
    assert result["intent"] in {"ask_characteristics", "select_shower_system"}
    assert "характерист" in result["answer"].lower() or "термостат" in result["answer"].lower()


# Проверяет, что бот умеет отвечать на нетоварную реплику.
def test_bot_uses_dialogue_answer_for_non_catalog_phrase() -> None:
    bot = PlumbingBot()
    result = bot.reply("ты бот")
    assert result["answer"]


# Проверяет интент самоидентификации бота.
def test_bot_handles_identity_intent() -> None:
    bot = PlumbingBot()
    result = bot.reply("ты бот")
    assert result["intent"] == "ask_bot_identity"
    assert "бот" in result["answer"].lower() or "консультант" in result["answer"].lower()


# Проверяет интент описания возможностей бота.
def test_bot_handles_capabilities_intent() -> None:
    bot = PlumbingBot()
    result = bot.reply("что ты умеешь")
    assert result["intent"] == "bot_capabilities"
    assert "могу" in result["answer"].lower() or "помога" in result["answer"].lower()


# Проверяет сбор комплекта товаров.
def test_bot_handles_complete_set_intent() -> None:
    bot = PlumbingBot()
    result = bot.reply("нужен комплект для ванной")
    assert result["intent"] == "select_complete_set"
    assert "комплект" in result["answer"].lower()


# Проверяет обычный small talk.
def test_bot_handles_small_talk() -> None:
    bot = PlumbingBot()
    result = bot.reply("Как у тебя дела?")
    assert result["intent"] == "small_talk"
    assert "хорош" in result["answer"].lower() or "спасибо" in result["answer"].lower()


# Проверяет, что приветствие не уходит в оффтоп.
def test_bot_handles_greeting_before_offtopic() -> None:
    bot = PlumbingBot()
    result = bot.reply("Привет")
    assert result["intent"] == "greeting"
    assert "привет" in result["answer"].lower() or "здрав" in result["answer"].lower()


# Проверяет retrieval-ответ из dialogues.txt.
def test_bot_uses_dialogues_for_free_topic_phrase() -> None:
    bot = PlumbingBot()
    result = bot.reply("Что хорошо?")
    assert result["intent"] == "small_talk"
    assert "горжусь" in result["answer"].lower() or "хорошо" in result["answer"].lower()


# Проверяет тематический fallback для свободной темы.
def test_bot_uses_thematic_dialogue_fallback() -> None:
    bot = PlumbingBot()
    result = bot.reply("как тебе спортивные машины")
    assert result["intent"] == "small_talk"
    assert "маш" in result["answer"].lower() or "автомоб" in result["answer"].lower() or "техник" in result["answer"].lower()


# Проверяет мягкий рекламный переход после small talk.
def test_bot_can_softly_transition_to_promo_after_small_talk() -> None:
    bot = PlumbingBot()
    bot.reply("Как у тебя дела?", conversation_id="chat-1")
    result = bot.reply("Что нового?", conversation_id="chat-1")
    assert result["intent"] == "small_talk"
    assert "кстати" in result["answer"].lower() or "каталог" in result["answer"].lower()


# Проверяет рекламу внутри сценария рекомендаций.
def test_bot_adds_soft_promo_to_recommendation() -> None:
    bot = PlumbingBot()
    result = bot.reply("что посоветуешь", conversation_id="chat-2")
    assert result["intent"] == "request_recommendation"
    assert "кстати" in result["answer"].lower() or "руб." in result["answer"].lower()


# Проверяет выбор категории систем монтажа.
def test_bot_handles_category_request_for_installation() -> None:
    bot = PlumbingBot()
    result = bot.reply("что есть из систем монтажа")
    assert "Системы монтажа" in result["answer"] or "SanFrame" in result["answer"]


# Проверяет выбор категории водонагревателей.
def test_bot_handles_category_request_for_water_heaters() -> None:
    bot = PlumbingBot()
    result = bot.reply("покажи водонагреватели")
    assert "HeatLine" in result["answer"] or "Водонагреватели" in result["answer"]


# Проверяет, что короткий recommendation-запрос не уходит в off-topic.
def test_bot_keeps_request_recommendation_for_short_prompt() -> None:
    bot = PlumbingBot()
    result = bot.reply("что посоветуешь")
    assert result["intent"] == "request_recommendation"
    assert "вариант" in result["answer"].lower() or "кстати" in result["answer"].lower()


# Проверяет, что продолжение рекомендаций остается в той же категории.
def test_bot_more_recommendations_stay_in_same_category_context() -> None:
    bot = PlumbingBot()
    bot.reply("покажи унитазы", conversation_id="chat-3")
    result = bot.reply("еще варианты", conversation_id="chat-3")
    assert "основные варианты" in result["answer"].lower()
    assert "AquaMix" not in result["answer"]


# Проверяет, что опечатка в товарном запросе не уводит бот в small talk.
def test_bot_handles_product_typo_in_runtime_flow() -> None:
    bot = PlumbingBot()
    result = bot.reply("нужен унитас")
    assert result["intent"] == "select_toilet"
    assert "унитаз" in result["answer"].lower() or "cerama" in result["answer"].lower()
