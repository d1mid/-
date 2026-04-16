from __future__ import annotations

from pathlib import Path

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None

from src.bot.utils.text import build_domain_vocabulary, normalize_text


DEFAULT_MODEL_PATH = Path("models/intent_model.joblib")


def load_model_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    if joblib is None:
        raise RuntimeError("Для предсказания нужен joblib. Установите зависимости из requirements.txt.")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Модель интентов не найдена: {model_path}")
    return joblib.load(model_path)


def predict_intent(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
    bundle = load_model_bundle(model_path)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    vocabulary = build_domain_vocabulary()
    normalized = normalize_text(text, vocabulary=vocabulary, mode="soft")
    return str(classifier.predict(vectorizer.transform([normalized]))[0])


def predict_intent_with_debug(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, str]:
    bundle = load_model_bundle(model_path)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    vocabulary = build_domain_vocabulary()
    normalized = normalize_text(text, vocabulary=vocabulary, mode="soft")
    intent = str(classifier.predict(vectorizer.transform([normalized]))[0])
    return {
        "original_text": text,
        "normalized_text": normalized,
        "intent": intent,
    }
