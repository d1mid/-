from __future__ import annotations

from pathlib import Path

from src.bot.utils.text import build_domain_vocabulary, levenshtein_distance, normalize_text


DEFAULT_DIALOGUES_PATH = Path("data/dialogues.txt")


def load_dialogue_pairs(dialogues_path: str | Path = DEFAULT_DIALOGUES_PATH) -> list[tuple[str, str]]:
    dialogues_path = Path(dialogues_path)
    content = dialogues_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    pairs: list[tuple[str, str]] = []
    seen_questions: set[str] = set()
    vocabulary = build_domain_vocabulary()

    for chunk in chunks:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        question = lines[0].removeprefix("-").strip()
        answer = lines[1].removeprefix("-").strip()
        normalized_question = normalize_text(question, vocabulary=vocabulary)
        if normalized_question and normalized_question not in seen_questions:
            seen_questions.add(normalized_question)
            pairs.append((normalized_question, answer))
    return pairs


def find_dialogue_answer(
    replica: str,
    dialogues_path: str | Path = DEFAULT_DIALOGUES_PATH,
) -> str | None:
    vocabulary = build_domain_vocabulary()
    normalized_replica = normalize_text(replica, vocabulary=vocabulary)
    if not normalized_replica:
        return None

    candidates: list[tuple[float, str]] = []
    for question, answer in load_dialogue_pairs(dialogues_path):
        if not question:
            continue
        length_gap = abs(len(normalized_replica) - len(question)) / max(len(question), 1)
        if length_gap >= 0.35:
            continue
        distance = levenshtein_distance(normalized_replica, question)
        weighted = distance / max(len(question), 1)
        if weighted <= 0.35:
            candidates.append((weighted, answer))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]
