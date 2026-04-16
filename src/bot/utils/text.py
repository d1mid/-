from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable

try:
    from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
except ImportError:  # pragma: no cover - optional dependency
    Doc = None
    MorphVocab = None
    NewsEmbedding = None
    NewsMorphTagger = None
    Segmenter = None


DEFAULT_PRODUCTS_PATH = Path("data/products.json")
DEFAULT_INTENTS_PATH = Path("data/intents.json")

WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ\s\-.,?!]")

RUSSIAN_STOPWORDS = {
    "а",
    "без",
    "более",
    "бы",
    "в",
    "во",
    "вот",
    "вы",
    "где",
    "да",
    "для",
    "до",
    "же",
    "за",
    "и",
    "из",
    "или",
    "к",
    "как",
    "какая",
    "какие",
    "какой",
    "ли",
    "мне",
    "можно",
    "мой",
    "моя",
    "на",
    "над",
    "надо",
    "не",
    "но",
    "ну",
    "о",
    "об",
    "от",
    "очень",
    "по",
    "под",
    "пожалуйста",
    "при",
    "про",
    "с",
    "со",
    "так",
    "там",
    "то",
    "у",
    "уже",
    "что",
    "эта",
    "этот",
    "эту",
    "я",
}

TOKEN_REPLACEMENTS = {
    "бойлера": "бойлер",
    "бойлеры": "бойлер",
    "ванную": "ванная",
    "ванной": "ванная",
    "ванне": "ванная",
    "водогрей": "водонагреватель",
    "водонагревателя": "водонагреватель",
    "водонагреватели": "водонагреватель",
    "душевую": "душевой",
    "душевыми": "душевой",
    "инсталляции": "инсталляция",
    "инсталляцию": "инсталляция",
    "крана": "кран",
    "краны": "кран",
    "кухн": "кухня",
    "кухне": "кухня",
    "кухню": "кухня",
    "мойку": "мойка",
    "подвесного": "подвесной",
    "подвесным": "подвесной",
    "раковине": "раковина",
    "раковину": "раковина",
    "раковины": "раковина",
    "раковин": "раковина",
    "скока": "сколько",
    "скок": "сколько",
    "смесителя": "смеситель",
    "смесители": "смеситель",
    "смесителья": "смеситель",
    "умывальник": "раковина",
    "умывальника": "раковина",
    "унитаза": "унитаз",
    "унитазов": "унитаз",
    "фильтром": "фильтр",
}

STEM_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ими",
    "ыми",
    "иях",
    "ах",
    "ях",
    "ия",
    "ья",
    "ов",
    "ев",
    "ий",
    "ый",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ам",
    "ям",
    "ом",
    "ем",
    "ы",
    "и",
    "а",
    "я",
    "е",
    "у",
    "ю",
)

SENTIMENT_LEXICON = {
    "люблю": 1.0,
    "нравится": 0.8,
    "отлично": 0.9,
    "хорошо": 0.6,
    "удобно": 0.5,
    "супер": 1.0,
    "отвратительно": -1.0,
    "плохо": -0.8,
    "дорого": -0.4,
    "ненавижу": -1.0,
    "ужасно": -0.9,
    "проблема": -0.5,
}

TOPIC_KEYWORDS = {
    "catalog": {"каталог", "ассортимент", "товар", "сантехника"},
    "kitchen": {"кухня", "мойка", "смеситель", "фильтр"},
    "bathroom": {"ванная", "раковина", "смеситель", "душевой"},
    "sanitary": {"унитаз", "инсталляция", "санузел"},
    "heating": {"водонагреватель", "бойлер", "литр", "нагрев"},
    "promo": {"акция", "выгодно", "рекомендуете", "лучше", "топ"},
    "price": {"цена", "стоимость", "рубль", "бюджет", "дорого"},
}


@dataclass(slots=True)
class TextProcessingResult:
    original: str
    cleaned: str
    corrected: str
    normalized: str
    original_tokens: list[str]
    corrected_tokens: list[str]
    normalized_tokens: list[str]
    corrections: dict[str, str]
    entities: dict[str, list[str]] = field(default_factory=dict)
    sentiment_label: str = "neutral"
    sentiment_score: float = 0.0
    topic: str = "general"
    natasha_used: bool = False


def clean_text(text: str) -> str:
    cleaned = text.replace("ё", "е").replace("Ё", "Е")
    cleaned = NON_TEXT_RE.sub(" ", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def strip_punctuation(text: str) -> str:
    return " ".join(tokenize(text))


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def max_typo_distance(token: str) -> int:
    if len(token) <= 4:
        return 1
    if len(token) <= 8:
        return 2
    return 3


def simple_stem(token: str) -> str:
    if token in TOKEN_REPLACEMENTS:
        return TOKEN_REPLACEMENTS[token]

    if len(token) <= 4:
        return token

    for suffix in STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _natasha_pipeline():
    if not all((Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter)):
        return None

    if not hasattr(_natasha_pipeline, "_cache"):
        try:
            embedding = NewsEmbedding()
            setattr(
                _natasha_pipeline,
                "_cache",
                {
                    "segmenter": Segmenter(),
                    "morph_tagger": NewsMorphTagger(embedding),
                    "morph_vocab": MorphVocab(),
                },
            )
        except Exception:  # pragma: no cover - environment-dependent fallback
            setattr(_natasha_pipeline, "_cache", None)
    return getattr(_natasha_pipeline, "_cache")


def natasha_available() -> bool:
    return _natasha_pipeline() is not None


def normalize_token(token: str) -> str:
    token = token.lower().replace("ё", "е")
    token = TOKEN_REPLACEMENTS.get(token, token)
    token = simple_stem(token)
    return TOKEN_REPLACEMENTS.get(token, token)


def lemmatize_text(text: str) -> tuple[list[str], bool]:
    pipeline = _natasha_pipeline()
    if pipeline is None:
        lemmas = [normalize_token(token) for token in tokenize(text)]
        return [lemma for lemma in lemmas if lemma], False

    doc = Doc(text)
    doc.segment(pipeline["segmenter"])
    doc.tag_morph(pipeline["morph_tagger"])

    lemmas: list[str] = []
    for token in doc.tokens:
        token.lemmatize(pipeline["morph_vocab"])
        lemma = token.lemma.lower().replace("ё", "е")
        lemma = TOKEN_REPLACEMENTS.get(lemma, lemma)
        if lemma:
            lemmas.append(lemma)
    return lemmas, True


def _iter_catalog_texts(products_path: Path) -> Iterable[str]:
    if not products_path.exists():
        return []

    with products_path.open("r", encoding="utf-8") as file:
        products = json.load(file)

    texts: list[str] = []
    if isinstance(products, list):
        for product in products:
            if isinstance(product, dict):
                texts.extend(_product_to_texts(product))
    elif isinstance(products, dict):
        for key in ("products", "featured_products", "categories"):
            value = products.get(key, [])
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        texts.extend(_product_to_texts(item))
                    elif isinstance(item, str):
                        texts.append(item)
    return texts


def _product_to_texts(product: dict) -> list[str]:
    texts: list[str] = []
    for key in ("id", "name", "category", "subcategory", "purpose", "promo_reason"):
        value = product.get(key)
        if isinstance(value, str):
            texts.append(value)
    for key in ("characteristics", "advantages"):
        value = product.get(key, [])
        if isinstance(value, list):
            texts.extend(item for item in value if isinstance(item, str))
    return texts


def _iter_intent_texts(intents_path: Path) -> Iterable[str]:
    if not intents_path.exists():
        return []

    with intents_path.open("r", encoding="utf-8") as file:
        intents_data = json.load(file)

    texts: list[str] = []
    for intent in intents_data.get("intents", []):
        if not isinstance(intent, dict):
            continue
        intent_name = intent.get("intent")
        if isinstance(intent_name, str):
            texts.append(intent_name.replace("_", " "))
        texts.extend(item for item in intent.get("examples", []) if isinstance(item, str))
        texts.extend(item for item in intent.get("responses", []) if isinstance(item, str))
    return texts


def build_domain_vocabulary(
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
    intents_path: str | Path = DEFAULT_INTENTS_PATH,
    extra_texts: Iterable[str] | None = None,
) -> set[str]:
    products_path = Path(products_path)
    intents_path = Path(intents_path)

    vocabulary: set[str] = set()
    sources = list(_iter_catalog_texts(products_path)) + list(_iter_intent_texts(intents_path))
    if extra_texts:
        sources.extend(extra_texts)

    for text in sources:
        cleaned = clean_text(text)
        for token in tokenize(cleaned):
            if len(token) >= 2:
                vocabulary.add(token)
                vocabulary.add(normalize_token(token))
        lemmas, _ = lemmatize_text(cleaned)
        for lemma in lemmas:
            if len(lemma) >= 2:
                vocabulary.add(lemma)

    vocabulary.update(TOKEN_REPLACEMENTS)
    vocabulary.update(TOKEN_REPLACEMENTS.values())
    vocabulary.update(
        {
            "ассортимент",
            "бюджет",
            "ванная",
            "водонагреватель",
            "доставка",
            "душ",
            "душевой",
            "инсталляция",
            "каталог",
            "комплект",
            "кухня",
            "лейка",
            "мойка",
            "подбор",
            "подвесной",
            "помочь",
            "помощь",
            "раковина",
            "ремонт",
            "санузел",
            "сантехника",
            "смеситель",
            "стойка",
            "термостат",
            "товар",
            "унитаз",
            "фильтр",
            "цена",
        }
    )
    return {token for token in vocabulary if token}


def correct_token(token: str, vocabulary: Iterable[str]) -> str:
    token = token.lower()
    vocabulary_set = set(vocabulary)
    if token in vocabulary_set or len(token) <= 2 or token.isdigit():
        return token

    best_match = token
    best_distance = max_typo_distance(token) + 1
    normalized_target = normalize_token(token)

    for candidate in vocabulary_set:
        if abs(len(candidate) - len(token)) > max_typo_distance(token):
            continue
        if candidate[:1] != token[:1] and normalize_token(candidate)[:1] != normalized_target[:1]:
            continue
        distance = levenshtein_distance(token, candidate)
        if distance < best_distance:
            best_match = candidate
            best_distance = distance
            if distance == 1:
                break

    return best_match if best_distance <= max_typo_distance(token) else token


def correct_typos(text: str, vocabulary: Iterable[str]) -> tuple[str, dict[str, str]]:
    tokens = tokenize(text)
    corrected_tokens: list[str] = []
    corrections: dict[str, str] = {}

    for token in tokens:
        corrected = correct_token(token, vocabulary)
        corrected_tokens.append(corrected)
        if corrected != token:
            corrections[token] = corrected

    return " ".join(corrected_tokens), corrections


def analyze_sentiment(tokens: Iterable[str]) -> tuple[str, float]:
    scores = [SENTIMENT_LEXICON[token] for token in tokens if token in SENTIMENT_LEXICON]
    if not scores:
        return "neutral", 0.0

    score = round(sum(scores) / len(scores), 3)
    if score > 0.2:
        return "positive", score
    if score < -0.2:
        return "negative", score
    return "neutral", score


def classify_topic(tokens: Iterable[str]) -> str:
    token_set = set(tokens)
    if not token_set:
        return "general"

    best_topic = "general"
    best_score = 0
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = len(token_set & keywords)
        if score > best_score:
            best_topic = topic
            best_score = score
    return best_topic


def extract_entities(
    text: str,
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
) -> dict[str, list[str]]:
    entities = {
        "product_ids": [],
        "product_names": [],
        "categories": [],
        "numbers": [],
    }

    cleaned = clean_text(text)
    lowered = cleaned.lower()
    tokens = tokenize(cleaned)
    query_lemmas, _ = lemmatize_text(cleaned)
    normalized_query_tokens = {
        TOKEN_REPLACEMENTS.get(lemma, lemma)
        for lemma in query_lemmas
        if lemma and lemma not in RUSSIAN_STOPWORDS
    }
    entities["numbers"] = [token for token in tokens if token.isdigit()]

    products_path = Path(products_path)
    if products_path.exists():
        products = json.loads(products_path.read_text(encoding="utf-8"))
        if isinstance(products, list):
            for product in products:
                product_id = str(product.get("id", "")).lower()
                product_name = str(product.get("name", "")).lower()
                category = str(product.get("category", "")).lower()

                if product_id and product_id in lowered:
                    entities["product_ids"].append(product.get("id"))
                product_lemmas, _ = lemmatize_text(product_name)
                normalized_product_tokens = {
                    TOKEN_REPLACEMENTS.get(lemma, lemma)
                    for lemma in product_lemmas
                    if lemma and lemma not in RUSSIAN_STOPWORDS
                }

                if product_name and product_name in lowered:
                    entities["product_names"].append(product.get("name"))
                elif normalized_product_tokens and normalized_product_tokens.issubset(normalized_query_tokens):
                    entities["product_names"].append(product.get("name"))
                if category and any(token == category or token in category for token in tokens):
                    entities["categories"].append(product.get("category"))

    entities["product_ids"] = sorted(set(filter(None, entities["product_ids"])))
    entities["product_names"] = sorted(set(filter(None, entities["product_names"])))
    entities["categories"] = sorted(set(filter(None, entities["categories"])))
    return entities


def normalize_text(text: str, vocabulary: Iterable[str] | None = None) -> str:
    result = preprocess_user_text(text, vocabulary=vocabulary)
    return result.normalized


def preprocess_user_text(text: str, vocabulary: Iterable[str] | None = None) -> TextProcessingResult:
    cleaned = clean_text(text)
    original_tokens = tokenize(cleaned)

    if vocabulary:
        corrected, corrections = correct_typos(cleaned, vocabulary)
    else:
        corrected = strip_punctuation(cleaned)
        corrections = {}

    corrected_tokens = tokenize(corrected)
    lemmas, natasha_used = lemmatize_text(corrected)
    normalized_tokens = [
        TOKEN_REPLACEMENTS.get(lemma, lemma)
        for lemma in lemmas
        if lemma and lemma not in RUSSIAN_STOPWORDS
    ]

    sentiment_label, sentiment_score = analyze_sentiment(normalized_tokens)
    topic = classify_topic(normalized_tokens)
    entities = extract_entities(corrected)

    return TextProcessingResult(
        original=text,
        cleaned=cleaned,
        corrected=corrected,
        normalized=" ".join(normalized_tokens),
        original_tokens=original_tokens,
        corrected_tokens=corrected_tokens,
        normalized_tokens=normalized_tokens,
        corrections=corrections,
        entities=entities,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        topic=topic,
        natasha_used=natasha_used,
    )
