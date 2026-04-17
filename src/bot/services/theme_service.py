"""Общая логика разговорных тем и базового определения сантехнического домена."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable


DEFAULT_THEME_KEYWORDS_PATH = Path("data/theme_keywords.json")

PLUMBING_DOMAIN_KEYWORDS = (
    "смес",
    "душ",
    "унитаз",
    "инстал",
    "монтаж",
    "раков",
    "водонагрев",
    "бойлер",
    "санфаян",
    "сантех",
    "кухн",
    "ванн",
    "сануз",
    "каталог",
    "товар",
    "цена",
    "бюджет",
    "руб",
    "ремонт",
    "фильтр",
)


# Загружает конфигурацию разговорных тем из JSON-файла.
@lru_cache(maxsize=2)
def load_theme_config(theme_keywords_path: str | Path = DEFAULT_THEME_KEYWORDS_PATH) -> dict[str, dict[str, tuple[str, ...]]]:
    path = Path(theme_keywords_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    themes = data.get("themes", {})
    config: dict[str, dict[str, tuple[str, ...]]] = {}
    for theme_name, theme_data in themes.items():
        config[theme_name] = {
            "keywords": tuple(theme_data.get("keywords", [])),
            "responses": tuple(theme_data.get("responses", [])),
        }
    return config


# Определяет наиболее подходящую тему по списку токенов.
def detect_theme_from_tokens(
    tokens: Iterable[str],
    theme_keywords_path: str | Path = DEFAULT_THEME_KEYWORDS_PATH,
) -> str | None:
    token_list = list(tokens)
    if not token_list:
        return None

    best_theme = None
    best_score = 0
    for theme, theme_data in load_theme_config(theme_keywords_path).items():
        keywords = theme_data.get("keywords", ())
        score = sum(1 for keyword in keywords if _contains_theme_token(token_list, keyword))
        if score > best_score:
            best_theme = theme
            best_score = score
    return best_theme


# Возвращает набор готовых ответов для выбранной темы.
def get_theme_responses(
    theme: str,
    theme_keywords_path: str | Path = DEFAULT_THEME_KEYWORDS_PATH,
) -> tuple[str, ...]:
    return load_theme_config(theme_keywords_path).get(theme, {}).get("responses", ())


# Определяет, относится ли запрос к сантехническому домену.
def is_plumbing_domain_text(
    text: str,
    normalized_tokens: Iterable[str] | None = None,
    entities: dict[str, list[str]] | None = None,
) -> bool:
    if entities and any(entities.get(key) for key in ("product_ids", "product_names", "categories")):
        return True

    lowered = text.lower()
    if any(keyword in lowered for keyword in PLUMBING_DOMAIN_KEYWORDS):
        return True

    token_list = list(normalized_tokens or [])
    return any(any(token == keyword or token.startswith(keyword) for token in token_list) for keyword in PLUMBING_DOMAIN_KEYWORDS)


# Проверяет, совпадает ли токен с ключом темы напрямую или по префиксу.
def _contains_theme_token(tokens: list[str], keyword: str) -> bool:
    return any(token == keyword or (len(keyword) >= 6 and token.startswith(keyword)) for token in tokens)
