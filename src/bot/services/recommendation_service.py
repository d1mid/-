"""Подбор товаров по интенту, бюджету и совпадениям с текстом запроса."""

from __future__ import annotations

from src.bot.services.catalog_service import extract_budget
from src.bot.utils.text import preprocess_user_text


INTENT_TO_FILTERS = {
    "select_kitchen_faucet": {"category": "Смесители", "subcategory": "Для кухни"},
    "select_bathroom_faucet": {"category": "Смесители", "subcategory": "Для раковины"},
    "select_shower_system": {"category": "Душевое оборудование"},
    "select_toilet": {"subcategory": "Унитазы"},
    "select_installation": {"subcategory": "Инсталляции"},
    "select_sink": {"subcategory": "Раковины"},
    "select_water_heater": {"category": "Водонагреватели"},
}


# Проверяет, подходит ли товар под набор точных фильтров.
def _matches_filters(product: dict, filters: dict[str, str]) -> bool:
    for key, value in filters.items():
        if product.get(key) != value:
            return False
    return True


# Вычисляет простой score товара по совпадениям с запросом и бюджету.
def _score_product(product: dict, normalized_query: str, budget: int | None) -> int:
    score = 0
    # Один общий текст удобен для простого rule-based скоринга без отдельных индексов.
    haystack = " ".join(
        [
            product.get("name", ""),
            product.get("category", ""),
            product.get("subcategory", ""),
            product.get("purpose", ""),
            " ".join(product.get("characteristics", [])),
            " ".join(product.get("advantages", [])),
        ]
    ).lower()

    for token in normalized_query.split():
        if token in haystack:
            score += 3

    if budget is not None:
        price = product.get("price_rub", 0)
        if price <= budget:
            score += 5
            score += max(0, 3 - abs(budget - price) // 3000)
        else:
            score -= 5

    if product.get("is_promoted"):
        score += 1
    return score


# Возвращает список наиболее подходящих товаров для пользовательского запроса.
def recommend_products(
    query: str,
    products: list[dict],
    intent: str | None = None,
    limit: int = 3,
) -> list[dict]:
    processed = preprocess_user_text(query)
    budget = extract_budget(query)

    filters: dict[str, str] = {}
    if intent in INTENT_TO_FILTERS:
        filters.update(INTENT_TO_FILTERS[intent])

    if "product_names" in processed.entities and processed.entities["product_names"]:
        return [product for product in products if product.get("name") in processed.entities["product_names"]][:limit]

    candidates = products
    if filters:
        filtered = [product for product in candidates if _matches_filters(product, filters)]
        if filtered:
            candidates = filtered

    if budget is not None:
        budget_filtered = [product for product in candidates if product.get("price_rub", 0) <= budget]
        if budget_filtered:
            candidates = budget_filtered

    scored = sorted(
        candidates,
        # Сначала берем более релевантные товары, потом используем цену как дополнительный порядок.
        key=lambda product: (_score_product(product, processed.normalized, budget), -(product.get("price_rub", 0))),
        reverse=True,
    )
    return scored[:limit]
