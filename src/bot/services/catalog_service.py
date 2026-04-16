from __future__ import annotations

import json
from pathlib import Path

from src.bot.utils.text import normalize_text


DEFAULT_PRODUCTS_PATH = Path("data/products.json")
DEFAULT_AD_SCENARIOS_PATH = Path("data/ad_scenarios.json")

CATEGORY_ALIASES = {
    "Смесители": [
        "смесители",
        "смеситель",
        "краны",
        "кран",
    ],
    "Душевое оборудование": [
        "душевое оборудование",
        "душевые",
        "душевая система",
        "душевые системы",
        "душевая стойка",
        "душевые стойки",
        "душ",
    ],
    "Санфаянс": [
        "санфаянс",
        "санфаянсу",
        "сантехника для санузла",
        "санузел",
    ],
    "Системы монтажа": [
        "системы монтажа",
        "систем монтажа",
        "система монтажа",
        "монтажные системы",
        "монтажная система",
        "монтажная рама",
        "монтажные рамы",
        "инсталляция",
        "инсталляции",
    ],
    "Водонагреватели": [
        "водонагреватели",
        "водонагреватель",
        "бойлер",
        "бойлеры",
    ],
}

SUBCATEGORY_ALIASES = {
    "Для кухни": ["для кухни", "кухонные", "кухня"],
    "Для раковины": ["для раковины", "для умывальника", "умывальник", "раковина"],
    "Душевая система": ["душевая система", "душевые системы", "душевая стойка"],
    "Инсталляции": ["инсталляции", "инсталляция", "система монтажа", "монтажная рама"],
    "Накопительные": ["накопительные", "накопительный", "бойлер"],
    "Раковины": ["раковины", "раковина"],
    "Унитазы": ["унитазы", "унитаз"],
}


def load_catalog(products_path: str | Path = DEFAULT_PRODUCTS_PATH) -> list[dict]:
    products_path = Path(products_path)
    if not products_path.exists():
        return []
    data = json.loads(products_path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("products", [])


def load_ad_scenarios(ad_scenarios_path: str | Path = DEFAULT_AD_SCENARIOS_PATH) -> list[dict]:
    ad_scenarios_path = Path(ad_scenarios_path)
    if not ad_scenarios_path.exists():
        return []
    data = json.loads(ad_scenarios_path.read_text(encoding="utf-8"))
    return data.get("scenarios", [])


def get_catalog_categories(products: list[dict]) -> list[str]:
    return sorted({product.get("category", "") for product in products if product.get("category")})


def find_category_in_query(query: str) -> str | None:
    lowered = query.lower()
    normalized = normalize_text(query, mode="catalog")
    for category, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            alias_normalized = normalize_text(alias, mode="catalog")
            if alias in lowered or (alias_normalized and alias_normalized in normalized):
                return category
    return None


def find_subcategory_in_query(query: str) -> str | None:
    lowered = query.lower()
    normalized = normalize_text(query, mode="catalog")
    for subcategory, aliases in SUBCATEGORY_ALIASES.items():
        for alias in aliases:
            alias_normalized = normalize_text(alias, mode="catalog")
            if alias in lowered or (alias_normalized and alias_normalized in normalized):
                return subcategory
    return None


def find_products_by_category(query: str, products: list[dict], limit: int = 5) -> list[dict]:
    category = find_category_in_query(query)
    subcategory = find_subcategory_in_query(query)

    matches = products
    if category:
        category_matches = [product for product in matches if product.get("category") == category]
        if category_matches:
            matches = category_matches
    if subcategory:
        subcategory_matches = [product for product in matches if product.get("subcategory") == subcategory]
        if subcategory_matches:
            matches = subcategory_matches

    return matches[:limit] if (category or subcategory) else []


def get_promoted_products(products: list[dict]) -> list[dict]:
    promoted = [product for product in products if product.get("is_promoted")]
    return sorted(promoted, key=lambda item: item.get("promo_priority") or 999)


def get_product_by_id(product_id: str, products: list[dict]) -> dict | None:
    product_id = product_id.lower()
    for product in products:
        if str(product.get("id", "")).lower() == product_id:
            return product
    return None


def find_products_by_name(query: str, products: list[dict]) -> list[dict]:
    normalized_query = normalize_text(query, mode="catalog")
    matches: list[dict] = []
    for product in products:
        product_name = product.get("name", "")
        normalized_name = normalize_text(product_name, mode="catalog")
        if normalized_query and (
            normalized_query in normalized_name
            or normalized_name in normalized_query
            or product_name.lower() in query.lower()
        ):
            matches.append(product)
    return matches


def format_product_brief(product: dict) -> str:
    return f"{product['name']} — {product['price_rub']} руб. {product['purpose']}"


def format_product_details(product: dict) -> str:
    characteristics = ", ".join(product.get("characteristics", [])[:5])
    advantages = ", ".join(product.get("advantages", [])[:3])
    return (
        f"{product['name']} стоит {product['price_rub']} руб. "
        f"Назначение: {product['purpose']} "
        f"Характеристики: {characteristics}. "
        f"Преимущества: {advantages}."
    )


def extract_budget(text: str) -> int | None:
    digits = []
    current = ""
    for char in text:
        if char.isdigit():
            current += char
        elif current:
            digits.append(current)
            current = ""
    if current:
        digits.append(current)

    if not digits:
        return None
    try:
        return int(max(digits, key=len))
    except ValueError:
        return None
