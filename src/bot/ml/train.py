from __future__ import annotations

import json
from pathlib import Path

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
except ImportError:  # pragma: no cover - optional dependency
    joblib = None
    TfidfVectorizer = None
    LinearSVC = None

from src.bot.utils.text import build_domain_vocabulary, normalize_text


DEFAULT_INTENTS_PATH = Path("data/intents.json")
DEFAULT_MODEL_PATH = Path("models/intent_model.joblib")


def load_intent_dataset(intents_path: str | Path = DEFAULT_INTENTS_PATH) -> tuple[list[str], list[str]]:
    intents_path = Path(intents_path)
    data = json.loads(intents_path.read_text(encoding="utf-8"))
    vocabulary = build_domain_vocabulary()

    texts: list[str] = []
    labels: list[str] = []
    for intent in data.get("intents", []):
        intent_name = intent["intent"]
        for example in intent.get("examples", []):
            texts.append(normalize_text(example, vocabulary=vocabulary))
            labels.append(intent_name)
    return texts, labels


def train_intent_model(
    intents_path: str | Path = DEFAULT_INTENTS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, str]:
    if TfidfVectorizer is None or LinearSVC is None or joblib is None:
        raise RuntimeError(
            "Для обучения нужны зависимости scikit-learn и joblib. Установите их из requirements.txt."
        )

    texts, labels = load_intent_dataset(intents_path)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    features = vectorizer.fit_transform(texts)

    classifier = LinearSVC()
    classifier.fit(features, labels)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, model_path)

    return {
        "status": "ok",
        "model_path": str(model_path),
        "samples": str(len(texts)),
        "classes": str(len(set(labels))),
    }
