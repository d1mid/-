"""Поиск ответа по dialogues.txt и по тематическому fallback-слою."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random

from src.bot.services.theme_service import (
    DEFAULT_THEME_KEYWORDS_PATH,
    _contains_theme_token,
    detect_theme_from_tokens,
    get_theme_responses,
    load_theme_config,
)
from src.bot.utils.text import build_domain_vocabulary, levenshtein_distance, normalize_text


DEFAULT_DIALOGUES_PATH = Path("data/dialogues.txt")
GENERIC_DIALOGUE_TOKENS = {
    "а",
    "быть",
    "в",
    "во",
    "вообще",
    "вот",
    "вы",
    "да",
    "делать",
    "для",
    "еще",
    "и",
    "или",
    "как",
    "какой",
    "мне",
    "можно",
    "надо",
    "не",
    "ну",
    "о",
    "он",
    "она",
    "они",
    "подскажи",
    "подсказать",
    "рассказать",
    "пока",
    "почему",
    "привет",
    "про",
    "просто",
    "расскажи",
    "любить",
    "нравиться",
    "думать",
    "с",
    "скажи",
    "так",
    "ты",
    "у",
    "что",
    "это",
    "этот",
    "я",
}

# Загружает пары вопрос-ответ из файла dialogues.txt.
def load_dialogue_pairs(dialogues_path: str | Path = DEFAULT_DIALOGUES_PATH) -> list[tuple[str, str]]:
    return list(_load_dialogue_pairs_cached(str(Path(dialogues_path))))


# Кэширует разбор dialogues.txt в готовый список диалоговых пар.
@lru_cache(maxsize=4)
def _load_dialogue_pairs_cached(dialogues_path: str) -> tuple[tuple[str, str], ...]:
    dialogues_path = Path(dialogues_path)
    content = dialogues_path.read_text(encoding="utf-8").strip()
    if not content:
        return tuple()

    pairs: list[tuple[str, str]] = []
    seen_questions: set[str] = set()
    vocabulary = build_domain_vocabulary()

    if "%%" in content:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines:
            if "%%" not in line:
                continue
            question, answer = [part.strip(" -") for part in line.split("%%", maxsplit=1)]
            normalized_question = normalize_text(question, vocabulary=vocabulary, mode="soft")
            if normalized_question and answer and normalized_question not in seen_questions:
                seen_questions.add(normalized_question)
                pairs.append((normalized_question, answer))
        return tuple(pairs)

    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    for chunk in chunks:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        question = lines[0].removeprefix("-").strip()
        answer = lines[1].removeprefix("-").strip()
        normalized_question = normalize_text(question, vocabulary=vocabulary, mode="soft")
        if normalized_question and question and answer and normalized_question not in seen_questions:
            seen_questions.add(normalized_question)
            pairs.append((normalized_question, answer))
    return tuple(pairs)


# Ищет лучший ответ по близкой фразе из dialogues.txt.
def find_dialogue_answer(
    replica: str,
    dialogues_path: str | Path = DEFAULT_DIALOGUES_PATH,
) -> str | None:
    vocabulary = build_domain_vocabulary()
    normalized_replica = normalize_text(replica, vocabulary=vocabulary, mode="soft")
    if not normalized_replica:
        return None

    replica_tokens = _meaningful_tokens(normalized_replica)
    if not replica_tokens:
        return None

    # Сначала собираем мини-датасет кандидатов по словам, как в классическом retrieval-подходе.
    candidate_ids = _get_dialogue_candidate_ids(tuple(sorted(replica_tokens)), str(Path(dialogues_path)))
    if not candidate_ids:
        return None

    answers: list[tuple[float, str, str]] = []
    pairs = load_dialogue_pairs(dialogues_path)
    for pair_index in candidate_ids:
        if pair_index >= len(pairs):
            continue
        question, answer = pairs[pair_index]
        if not question:
            continue
        length_gap = abs(len(normalized_replica) - len(question)) / max(len(question), 1)
        if length_gap >= 0.25:
            continue
        question_tokens = _meaningful_tokens(question)
        if replica_tokens and question_tokens and not (replica_tokens & question_tokens):
            continue
        distance = levenshtein_distance(normalized_replica, question)
        weighted = distance / max(len(question), 1)
        if weighted <= 0.25:
            answers.append((weighted, question, answer))

    if not answers:
        return None
    return min(answers, key=lambda item: item[0])[2]


# Ищет тематический ответ, если подходящей близкой реплики не найдено.
def find_thematic_dialogue_answer(
    replica: str,
    dialogues_path: str | Path = DEFAULT_DIALOGUES_PATH,
    theme_keywords_path: str | Path = DEFAULT_THEME_KEYWORDS_PATH,
) -> str | None:
    normalized_replica = normalize_text(replica, mode="soft")
    if not normalized_replica:
        return None

    theme = detect_theme_from_tokens(normalized_replica.split(), theme_keywords_path)
    if not theme:
        return None

    answers = get_theme_responses(theme, theme_keywords_path) or _get_theme_answers(theme, str(Path(dialogues_path)), str(Path(theme_keywords_path)))
    if not answers:
        return None

    # Небольшая стабильность выбора без хранения отдельного состояния.
    seed = sum(ord(char) for char in normalized_replica)
    chooser = random.Random(seed)
    return chooser.choice(list(answers))


# Подбирает ответы из dialogues.txt, которые относятся к заданной теме.
@lru_cache(maxsize=64)
def _get_theme_answers(theme: str, dialogues_path: str, theme_keywords_path: str) -> tuple[str, ...]:
    keywords = load_theme_config(theme_keywords_path).get(theme, {}).get("keywords", ())
    if not keywords:
        return tuple()

    answers: list[str] = []
    seen_answers: set[str] = set()
    for question, answer in _load_dialogue_pairs_cached(dialogues_path):
        question_tokens = question.split()
        if any(_contains_theme_token(question_tokens, keyword) for keyword in keywords) and answer not in seen_answers:
            seen_answers.add(answer)
            answers.append(answer)
    return tuple(answers)
# Отбрасывает слишком общие слова, чтобы поиск работал по значимым токенам.
def _meaningful_tokens(text: str) -> set[str]:
    return {token for token in text.split() if token and token not in GENERIC_DIALOGUE_TOKENS}


# Возвращает индексы диалоговых пар-кандидатов по набору токенов запроса.
@lru_cache(maxsize=128)
def _get_dialogue_candidate_ids(tokens: tuple[str, ...], dialogues_path: str) -> tuple[int, ...]:
    if not tokens:
        return tuple()

    index = _load_dialogue_index_cached(dialogues_path)
    candidate_ids: set[int] = set()
    for token in tokens:
        candidate_ids.update(index.get(token, ()))
    return tuple(sorted(candidate_ids))


# Строит и кэширует индекс dialogues.txt по значимым словам.
@lru_cache(maxsize=4)
def _load_dialogue_index_cached(dialogues_path: str) -> dict[str, tuple[int, ...]]:
    pairs = _load_dialogue_pairs_cached(dialogues_path)
    index: dict[str, list[tuple[int, int]]] = {}

    for pair_index, (question, _) in enumerate(pairs):
        tokens = _meaningful_tokens(question)
        question_length = len(question)
        for token in tokens:
            index.setdefault(token, []).append((pair_index, question_length))

    compact_index: dict[str, tuple[int, ...]] = {}
    for token, candidates in index.items():
        # Храним только ограниченный список самых коротких кандидатов, чтобы поиск был быстрее.
        candidates.sort(key=lambda item: item[1])
        compact_index[token] = tuple(pair_index for pair_index, _ in candidates[:1000])
    return compact_index
