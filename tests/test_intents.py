"""Проверки роутинга интентов в наиболее уязвимых сценариях."""

from src.bot.core.bot import PlumbingBot


# Проверяет, что recommendation-интент не поглощается off-topic эвристикой.
def test_request_recommendation_intent_is_not_demoted_to_small_talk() -> None:
    bot = PlumbingBot()
    result = bot.reply("что посоветуешь")
    assert result["intent"] == "request_recommendation"


# Проверяет, что опечатка по доменному слову сохраняет товарный интент.
def test_typo_query_still_maps_to_domain_intent() -> None:
    bot = PlumbingBot()
    result = bot.reply("нужен унитас")
    assert result["intent"] == "select_toilet"
