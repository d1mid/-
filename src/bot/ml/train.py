"""Обучение модели интентов на примерах из intents.json."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import joblib
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
except ImportError:  # pragma: no cover - optional dependency
    joblib = None
    accuracy_score = None
    train_test_split = None
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
            # На обучении используем ту же мягкую нормализацию, что и на предсказании.
            texts.append(normalize_text(example, vocabulary=vocabulary, mode="soft"))
            labels.append(intent_name)
    return texts, labels


def train_intent_model(
    intents_path: str | Path = DEFAULT_INTENTS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, str | float]:
    if (
        TfidfVectorizer is None
        or LinearSVC is None
        or joblib is None
        or train_test_split is None
        or accuracy_score is None
    ):
        raise RuntimeError(
            "Для обучения нужны зависимости scikit-learn и joblib. Установите их из requirements.txt."
        )

    texts, labels = load_intent_dataset(intents_path)
    if not texts:
        raise ValueError("Датасет интентов пуст. Невозможно обучить модель.")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    features = vectorizer.fit_transform(texts)

    test_size = 0.2 if len(texts) >= 40 else 0.25
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    classifier = LinearSVC()
    classifier.fit(x_train, y_train)

    train_accuracy = accuracy_score(y_train, classifier.predict(x_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(x_test))

    classifier.fit(features, labels)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": sorted(set(labels)),
            "metrics": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "samples": len(texts),
                "classes": len(set(labels)),
            },
        },
        model_path,
    )

    return {
        "status": "ok",
        "model_path": str(model_path),
        "samples": len(texts),
        "classes": len(set(labels)),
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
    }


if __name__ == "__main__":
    result = train_intent_model()
    print(json.dumps(result, ensure_ascii=False, indent=2))
