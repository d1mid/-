"""Проверки рекомендаций и памяти о предыдущем списке."""

from src.bot.core.bot import PlumbingBot


# Проверяет, что "еще варианты" расширяет предыдущий список, а не перескакивает в другую категорию.
def test_follow_up_recommendations_keep_previous_context() -> None:
    bot = PlumbingBot()
    bot.reply("покажи унитазы", conversation_id="recommendation-chat")
    result = bot.reply("еще варианты", conversation_id="recommendation-chat")

    assert "основные варианты" in result["answer"].lower()
    assert "AquaMix" not in result["answer"]


# Проверяет, что после рекомендаций можно спросить цену у первого товара из того же списка.
def test_referenced_product_uses_last_recommendation_list() -> None:
    bot = PlumbingBot()
    first = bot.reply("покажи унитазы", conversation_id="price-chat")
    second = bot.reply("у первого какая цена", conversation_id="price-chat")

    assert "Cerama" in first["answer"]
    assert "Cerama Floor T-100" in second["answer"]
