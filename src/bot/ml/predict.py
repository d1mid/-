from __future__ import annotations

from pathlib import Path

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None

from src.bot.utils.text import build_domain_vocabulary, normalize_text


DEFAULT_MODEL_PATH = Path("models/intent_model.joblib")


def predict_intent(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
    if joblib is None:
        raise RuntimeError("Для предсказания нужен joblib. Установите зависимости из requirements.txt.")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Модель интентов не найдена: {model_path}")

    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    vocabulary = build_domain_vocabulary()
    normalized = normalize_text(text, vocabulary=vocabulary)
    return classifier.predict(vectorizer.transform([normalized]))[0]
