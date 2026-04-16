from src.bot.utils.text import (
    build_domain_vocabulary,
    clean_text,
    levenshtein_distance,
    normalize_text,
    preprocess_user_text,
    natasha_available,
)


def test_clean_text_removes_extra_symbols() -> None:
    assert clean_text("  Нужен@@@   смеситель!!!   ") == "Нужен смеситель!!!"


def test_levenshtein_distance_works_for_typo() -> None:
    assert levenshtein_distance("смиситель", "смеситель") == 1


def test_normalize_text_corrects_typos_with_domain_vocabulary() -> None:
    vocabulary = build_domain_vocabulary()
    normalized = normalize_text("Нужен смиситель для кухне", vocabulary=vocabulary)
    assert "смеситель" in normalized
    assert "кухня" in normalized


def test_preprocess_user_text_collects_corrections() -> None:
    vocabulary = build_domain_vocabulary()
    result = preprocess_user_text("Ищу унитас с микролифтом", vocabulary=vocabulary)

    assert result.corrected == "ищу унитаз с микролифтом"
    assert result.corrections["унитас"] == "унитаз"
    assert "унитаз" in result.normalized_tokens


def test_preprocess_user_text_keeps_numbers_for_budget_queries() -> None:
    vocabulary = build_domain_vocabulary()
    result = preprocess_user_text("Что есть до 15000 рублей?", vocabulary=vocabulary)

    assert "15000" in result.corrected_tokens


def test_preprocess_user_text_extracts_topic_and_entities() -> None:
    vocabulary = build_domain_vocabulary()
    result = preprocess_user_text("Сколько стоит AquaMix ProFilter K-500?", vocabulary=vocabulary)

    assert result.topic in {"price", "kitchen", "general"}
    assert "AquaMix ProFilter K-500" in result.entities["product_names"]


def test_natasha_flag_is_boolean() -> None:
    assert isinstance(natasha_available(), bool)
