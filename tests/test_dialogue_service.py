from src.bot.services.dialogue_service import find_dialogue_answer, find_thematic_dialogue_answer, load_dialogue_pairs


def test_load_dialogue_pairs_returns_dataset() -> None:
    pairs = load_dialogue_pairs()
    assert pairs
    assert any("смеситель" in question for question, _ in pairs)


def test_find_dialogue_answer_matches_typical_question() -> None:
    answer = find_dialogue_answer("нужен смиситель для кухни")
    assert answer is not None
    assert "кух" in answer.lower() or "смес" in answer.lower()


def test_find_thematic_dialogue_answer_for_cars() -> None:
    answer = find_thematic_dialogue_answer("как тебе спортивные машины")
    assert answer is not None
    assert "маш" in answer.lower() or "автомоб" in answer.lower() or "техник" in answer.lower()


def test_find_thematic_dialogue_answer_for_weather() -> None:
    answer = find_thematic_dialogue_answer("сегодня ужасно холодно и ветрено")
    assert answer is not None
    assert "погод" in answer.lower() or "терпен" in answer.lower() or "разговор" in answer.lower()
