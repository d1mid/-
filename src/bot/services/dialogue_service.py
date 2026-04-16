from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random

from src.bot.utils.text import build_domain_vocabulary, levenshtein_distance, normalize_text


DEFAULT_DIALOGUES_PATH = Path("data/dialogues.txt")

DIALOGUE_THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "greeting": ("привет", "здравствуйте", "добрый", "вечер", "утро", "день"),
    "mood": ("дело", "настроение", "жизнь", "новое", "сам"),
    "support": ("грустный", "тревога", "устать", "плохо", "одиноко", "поддержка", "страшно", "нервничать"),
    "weather": ("погода", "дождь", "снег", "ветер", "жарко", "холодный", "солнечно"),
    "plans": ("план", "сегодня", "вечер", "выходной", "заняться", "делать"),
    "food": ("еда", "поесть", "кофе", "чай", "голодный", "пицца", "сладкое", "готовить"),
    "music": ("музыка", "песня", "слушать", "рок", "рэп", "джаз", "концерт"),
    "movies": ("фильм", "сериал", "мультфильм", "аниме", "смотреть", "комедия", "драма", "ужасы"),
    "books": ("книга", "читать", "почитать", "детектив", "фантастика", "классика"),
    "games": ("игра", "играть", "стратегия", "шутер", "инди", "сюжетный"),
    "sport": ("спорт", "тренировка", "футбол", "баскетбол", "бег", "мотивация"),
    "travel": ("путешествие", "отпуск", "море", "гора", "поезд", "самолет", "природа"),
    "animals": ("животное", "кошка", "кот", "собака", "попугай", "рыбка", "лошадь"),
    "cars": ("машина", "авто", "bmw", "mercedes", "audi", "porsche", "порш", "мотоцикл"),
    "colors": ("цвет", "черный", "белый", "красный", "синий", "зеленый", "серый"),
    "home": ("уют", "порядок", "дом", "тишина", "шум", "ночь", "утро", "бардак"),
    "identity": ("кто", "бот", "робот", "человек", "настоящий", "живой", "умный"),
    "communication": ("вежливый", "грубый", "ответ", "ошибка", "понять", "просто", "коротко", "длинный"),
    "philosophy": ("смысл", "жизнь", "любовь", "дружба", "счастье", "успех", "философия"),
}

DIALOGUE_THEME_RESPONSES: dict[str, tuple[str, ...]] = {
    "greeting": (
        "Привет. Можно просто поговорить или перейти к любой теме, которая вам интересна.",
        "Здравствуйте. Я готов поддержать спокойный разговор.",
    ),
    "mood": (
        "Все нормально, спасибо. А у вас как настроение?",
        "Неплохо. Если хотите, можем просто немного поболтать.",
    ),
    "support": (
        "Понимаю. Иногда уже спокойный разговор немного помогает.",
        "Сочувствую. Давайте без спешки, можно просто немного поговорить.",
        "Если вам сейчас тяжело, лучше снизить темп и не давить на себя лишний раз.",
    ),
    "weather": (
        "С погодой редко получается договориться, но обсудить ее всегда можно.",
        "Такая погода обычно просто требует чуть больше терпения.",
        "Погода меняется, а разговор можно сделать спокойнее.",
    ),
    "plans": (
        "Если планов нет, это тоже иногда хороший план.",
        "Лучше всего выбирать что-то посильное, без лишней гонки.",
        "На вечер обычно хорошо работают простые вещи: отдых, музыка, фильм или прогулка.",
    ),
    "food": (
        "Еда и напитки — одна из самых надежных тем для мирного разговора.",
        "Если хочется чего-то приятного, чай или что-то вкусное уже неплохое начало.",
        "О еде можно говорить долго, особенно когда пора сделать паузу.",
    ),
    "music": (
        "Музыка обычно хорошо подстраивается под настроение, в этом ее сила.",
        "Если хочется спокойствия, лучше выбрать что-то мягкое и без перегруза.",
        "Хорошая музыка часто просто помогает выдохнуть.",
    ),
    "movies": (
        "Фильмы удобны тем, что быстро меняют настроение и фон вечера.",
        "Лучше выбирать фильм не по громкости названия, а по своему состоянию.",
        "Если день был тяжелым, обычно лучше что-то полегче.",
    ),
    "books": (
        "Книги хороши, когда хочется чуть больше тишины и внимания.",
        "Иногда лучше начать с небольшой и понятной книги, а не с чего-то тяжелого.",
        "Чтение хорошо работает, когда не хочется суеты.",
    ),
    "games": (
        "Игры для многих — это способ переключиться и снять напряжение.",
        "Тут все зависит от темпа: кому-то ближе сюжет, кому-то динамика.",
        "Лучше всего выбирать игру под настроение, а не просто по популярности.",
    ),
    "sport": (
        "Спорт многим помогает разгрузить голову и вернуть ощущение ритма.",
        "Главное в спорте — не идеальность, а регулярность и адекватная нагрузка.",
        "Даже небольшой старт обычно полезнее больших обещаний.",
    ),
    "travel": (
        "Путешествия часто нужны просто ради смены картинки и ритма.",
        "Отдых обычно удается лучше там, где не приходится спешить каждую минуту.",
        "Даже короткая поездка иногда хорошо перезагружает.",
    ),
    "animals": (
        "Животные часто делают разговор мягче уже самим своим присутствием.",
        "Кошки, собаки и другие животные обычно легко становятся отдельной теплой темой.",
        "О животных обычно приятно говорить, особенно если хочется чего-то спокойного.",
    ),
    "cars": (
        "Машины для многих — это не только транспорт, но и вкус, характер и стиль.",
        "У автомобильных тем обычно всегда есть свой азарт и свой язык.",
        "Если речь о машинах, люди часто спорят о вкусе не меньше, чем о технике.",
    ),
    "colors": (
        "С цветами все очень субъективно: важнее, чтобы оттенок не надоедал вам самому.",
        "Цвет часто задает настроение сильнее, чем кажется на первый взгляд.",
        "У каждого цвета свой характер, и в этом вся прелесть выбора.",
    ),
    "home": (
        "Дом и уют обычно строятся из мелочей, а не из громких решений.",
        "Порядок и спокойная атмосфера часто дают больше, чем идеальная картинка.",
        "Иногда лучше всего просто немного упростить пространство вокруг себя.",
    ),
    "identity": (
        "Я виртуальный собеседник, так что в основном существую в разговоре.",
        "Я не человек, но могу поддерживать беседу и быть полезным в диалоге.",
        "Я бот, но стараюсь отвечать спокойно и по-человечески.",
    ),
    "communication": (
        "Если ответ звучит неудачно, всегда можно переформулировать и сделать его проще.",
        "Нормальный разговор обычно держится на тоне не меньше, чем на смысле.",
        "Можно говорить короче, мягче или проще — это все настраивается.",
    ),
    "philosophy": (
        "На общие вопросы редко бывает один правильный ответ, но обсудить их всегда интересно.",
        "Философские темы хороши, пока не превращаются в сплошной туман.",
        "Иногда большой вопрос лучше разбирать по частям, а не штурмовать целиком.",
    ),
}


def load_dialogue_pairs(dialogues_path: str | Path = DEFAULT_DIALOGUES_PATH) -> list[tuple[str, str]]:
    return list(_load_dialogue_pairs_cached(str(Path(dialogues_path))))


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


def find_dialogue_answer(
    replica: str,
    dialogues_path: str | Path = DEFAULT_DIALOGUES_PATH,
) -> str | None:
    vocabulary = build_domain_vocabulary()
    normalized_replica = normalize_text(replica, vocabulary=vocabulary, mode="soft")
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


def find_thematic_dialogue_answer(
    replica: str,
    dialogues_path: str | Path = DEFAULT_DIALOGUES_PATH,
) -> str | None:
    normalized_replica = normalize_text(replica, mode="soft")
    if not normalized_replica:
        return None

    theme = _detect_dialogue_theme(normalized_replica)
    if not theme:
        return None

    answers = DIALOGUE_THEME_RESPONSES.get(theme) or _get_theme_answers(theme, str(Path(dialogues_path)))
    if not answers:
        return None

    # Небольшая стабильность выбора без хранения отдельного состояния.
    seed = sum(ord(char) for char in normalized_replica)
    chooser = random.Random(seed)
    return chooser.choice(list(answers))


def _detect_dialogue_theme(normalized_replica: str) -> str | None:
    tokens = normalized_replica.split()
    best_theme = None
    best_score = 0
    for theme, keywords in DIALOGUE_THEME_KEYWORDS.items():
        score = sum(1 for keyword in keywords if _contains_theme_token(tokens, keyword))
        if score > best_score:
            best_theme = theme
            best_score = score
    return best_theme


@lru_cache(maxsize=64)
def _get_theme_answers(theme: str, dialogues_path: str) -> tuple[str, ...]:
    keywords = DIALOGUE_THEME_KEYWORDS.get(theme, ())
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


def _contains_theme_token(tokens: list[str], keyword: str) -> bool:
    return any(token == keyword or (len(keyword) >= 6 and token.startswith(keyword)) for token in tokens)
