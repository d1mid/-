"""Загрузка обученной модели и предсказание интента по пользовательской фразе."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None

from src.bot.utils.text import build_domain_vocabulary, normalize_text


DEFAULT_MODEL_PATH = Path("models/intent_model.joblib")


# Загружает сохраненный bundle модели с векторизатором и классификатором.
def load_model_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    return _load_model_bundle_cached(str(Path(model_path)))


# Кэширует загрузку модели, чтобы не читать ее с диска при каждом запросе.
@lru_cache(maxsize=4)
def _load_model_bundle_cached(model_path: str) -> dict:
    if joblib is None:
        raise RuntimeError("Для предсказания нужен joblib. Установите зависимости из requirements.txt.")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Модель интентов не найдена: {path}")
    return joblib.load(path)


# Предсказывает интент для пользовательской фразы.
def predict_intent(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
    bundle = load_model_bundle(model_path)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]

    # Нормализация должна совпадать с логикой, использованной на обучении.
    vocabulary = build_domain_vocabulary()
    normalized = normalize_text(text, vocabulary=vocabulary, mode="soft")
    return str(classifier.predict(vectorizer.transform([normalized]))[0])
