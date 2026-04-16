from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
import re

from src.bot.ml.predict import load_model_bundle, predict_intent
from src.bot.services.catalog_service import (
    DEFAULT_AD_SCENARIOS_PATH,
    DEFAULT_PRODUCTS_PATH,
    extract_budget,
    find_category_in_query,
    find_products_by_name,
    find_products_by_category,
    format_product_brief,
    format_product_details,
    get_catalog_categories,
    get_product_by_id,
    get_promoted_products,
    load_ad_scenarios,
    load_catalog,
)
from src.bot.services.dialogue_service import find_dialogue_answer, find_thematic_dialogue_answer
from src.bot.services.dialogue_service import load_dialogue_pairs
from src.bot.services.recommendation_service import recommend_products
from src.bot.utils.text import _load_product_entity_index, build_domain_vocabulary, natasha_available, preprocess_user_text


DEFAULT_INTENTS_PATH = Path("data/intents.json")


@dataclass
class ConversationState:
    turns: int = 0
    free_talk_turns: int = 0
    promo_cooldown: int = 0
    last_intent: str = "fallback"
    last_reply_kind: str = "generic"
    last_recommendation_ids: list[str] | None = None


class PlumbingBot:
    DIRECT_INTENT_PRIORITY = {
        "greeting",
        "goodbye",
        "small_talk",
        "bot_capabilities",
        "ask_bot_identity",
        "request_recommendation",
        "select_complete_set",
        "ask_availability",
        "ask_delivery_installation",
    }
    NON_TARGET_INTENTS = {"small_talk", "bot_capabilities", "ask_bot_identity"}
    RETURN_TO_PLUMBING_PHRASES = (
        "Если захотите, можем перейти к сантехнике: покажу каталог или помогу с выбором.",
        "Если хотите вернуться к делу, я могу подобрать смеситель, душевую систему, раковину, унитаз или водонагреватель.",
        "Если захотите, могу сразу перейти к сантехнике и предложить варианты по бюджету или категории.",
        "Если хотите, можем переключиться на сантехнику: цену, характеристики, сравнение или подбор.",
    )

    FOLLOW_UP_RECOMMENDATION_RE = re.compile(
        r"^(а еще( что)?|что еще|еще что|еще варианты|другие варианты|еще|а еще есть)\??$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        products_path: str | Path = DEFAULT_PRODUCTS_PATH,
        intents_path: str | Path = DEFAULT_INTENTS_PATH,
        ad_scenarios_path: str | Path = DEFAULT_AD_SCENARIOS_PATH,
    ) -> None:
        self.products = load_catalog(products_path)
        self.intents = self._load_intents(intents_path)
        self.ad_scenarios = load_ad_scenarios(ad_scenarios_path)
        self.sessions: dict[str, ConversationState] = {}
        self._warm_up(products_path)

    @staticmethod
    def _warm_up(products_path: str | Path) -> None:
        try:
            build_domain_vocabulary()
            _load_product_entity_index(str(Path(products_path)))
            load_dialogue_pairs()
            load_model_bundle()
            if natasha_available():
                preprocess_user_text("тестовый прогрев")
        except Exception:
            # Если прогрев не удался, бот все равно продолжит работу через обычный путь.
            pass

    @staticmethod
    def _load_intents(intents_path: str | Path) -> dict[str, dict]:
        data = json.loads(Path(intents_path).read_text(encoding="utf-8"))
        return {item["intent"]: item for item in data.get("intents", [])}

    def _random_response(self, intent: str, fallback_intent: str = "fallback") -> str:
        responses = self.intents.get(intent, {}).get("responses")
        if responses:
            return random.choice(responses)
        return random.choice(self.intents.get(fallback_intent, {}).get("responses", ["Не совсем понял запрос."]))

    def _handle_show_catalog(self) -> str:
        categories = ", ".join(get_catalog_categories(self.products))
        return f"В каталоге есть следующие категории: {categories}."

    def _handle_category_request(self, text: str) -> tuple[str | None, list[dict]]:
        items = find_products_by_category(text, self.products, limit=5)
        category = find_category_in_query(text)
        if not items:
            return None, []

        prefix = "Вот товары из этой категории:"
        if category:
            prefix = f"Вот несколько товаров из категории «{category}»:"
        return prefix + "\n- " + "\n- ".join(format_product_brief(item) for item in items), items

    def _handle_product_price(self, text: str) -> str | None:
        matches = find_products_by_name(text, self.products)
        if matches:
            product = matches[0]
            return f"{product['name']} стоит {product['price_rub']} руб."
        return None

    def _handle_product_characteristics(self, text: str) -> str | None:
        matches = find_products_by_name(text, self.products)
        if matches:
            return format_product_details(matches[0])
        return None

    def _handle_referenced_product(self, text: str, state: ConversationState) -> dict | None:
        if not state.last_recommendation_ids:
            return None

        lowered = text.lower()
        ordinal_map = {
            0: ("перв", "1", "один"),
            1: ("втор", "2", "два"),
            2: ("трет", "3", "три"),
            3: ("четвер", "4", "четыр"),
            4: ("пят", "5", "пять"),
        }

        index = None
        for candidate_index, markers in ordinal_map.items():
            if any(marker in lowered for marker in markers):
                index = candidate_index
                break

        if index is None or index >= len(state.last_recommendation_ids):
            return None
        return get_product_by_id(state.last_recommendation_ids[index], self.products)

    def _handle_compare_products(self, text: str) -> str | None:
        matches = find_products_by_name(text, self.products)
        if len(matches) < 2:
            return "Для сравнения напишите две модели, и я покажу разницу по цене, назначению и характеристикам."

        left, right = matches[:2]
        return (
            f"{left['name']} — {left['price_rub']} руб., {left['purpose']} "
            f"{right['name']} — {right['price_rub']} руб., {right['purpose']} "
            f"Если хотите, я могу дальше сравнить их по характеристикам и преимуществам."
        )

    def _handle_recommendations(self, text: str, intent: str) -> str | None:
        items = recommend_products(text, self.products, intent=intent, limit=3)
        if not items:
            return None

        lines = [format_product_brief(item) for item in items]
        prefix = "Вот несколько подходящих вариантов:"
        return prefix + "\n- " + "\n- ".join(lines)

    def _get_more_recommendations(self, state: ConversationState) -> str | None:
        if not state.last_recommendation_ids:
            return None

        seen = set(state.last_recommendation_ids)
        items = [product for product in self.products if product.get("id") not in seen][:3]
        if not items:
            return "Пока это основные варианты, которые я могу предложить сразу. Если хотите, могу сузить выбор по бюджету, помещению или типу товара."

        state.last_recommendation_ids.extend([item["id"] for item in items])
        return "Можно посмотреть еще такие варианты:\n- " + "\n- ".join(format_product_brief(item) for item in items)

    def _handle_budget(self, text: str, intent: str) -> str | None:
        budget = extract_budget(text)
        if budget is None:
            return self._random_response("selection_by_budget")

        items = recommend_products(text, self.products, intent=intent, limit=3)
        if not items:
            return f"Пока не нашел подходящих товаров до {budget} руб. Могу предложить похожие варианты из соседнего ценового диапазона."
        lines = [format_product_brief(item) for item in items]
        return f"Под ваш бюджет до {budget} руб. подходят такие варианты:\n- " + "\n- ".join(lines)

    def _handle_promo(self, intent: str) -> str | None:
        matched = [scenario for scenario in self.ad_scenarios if intent in scenario.get("trigger_intents", [])]
        if not matched and intent == "promo_offer":
            matched = self.ad_scenarios[:3]
        if not matched:
            return None

        blocks = []
        for scenario in matched[:3]:
            product = get_product_by_id(scenario["product_id"], self.products)
            header = scenario["product_name"]
            if product:
                header = f"{scenario['product_name']} — {product['price_rub']} руб."
            blocks.append(header + ". " + " ".join(scenario.get("messages", [])[:2]))
        return "Рекомендую обратить внимание на такие предложения:\n- " + "\n- ".join(blocks)

    def _handle_showcase_promoted(self) -> str:
        items = get_promoted_products(self.products)[:3]
        if not items:
            return self._random_response("promo_offer")
        return "Сейчас могу порекомендовать такие модели:\n- " + "\n- ".join(format_product_brief(item) for item in items)

    def _handle_complete_set(self, text: str) -> str:
        lowered = text.lower()
        if "ван" in lowered:
            items = []
            for wanted_intent in ("select_sink", "select_bathroom_faucet", "select_shower_system"):
                items.extend(recommend_products(text, self.products, intent=wanted_intent, limit=1))
        elif "сануз" in lowered or "унитаз" in lowered:
            items = []
            for wanted_intent in ("select_toilet", "select_installation"):
                items.extend(recommend_products(text, self.products, intent=wanted_intent, limit=1))
        else:
            items = recommend_products(text, self.products, intent="select_for_renovation", limit=3)

        unique_items: list[dict] = []
        seen_ids: set[str] = set()
        for item in items:
            item_id = item.get("id")
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_items.append(item)

        if not unique_items:
            return self._random_response("select_complete_set")
        return "Могу предложить такой комплект:\n- " + "\n- ".join(format_product_brief(item) for item in unique_items[:3])

    @staticmethod
    def _looks_like_offtopic(text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return False

        domain_keywords = (
            "смес",
            "душ",
            "унитаз",
            "инстал",
            "монтаж",
            "раков",
            "водонагрев",
            "бойлер",
            "санфаян",
            "сантех",
            "кухн",
            "ванн",
            "сануз",
            "каталог",
            "товар",
            "цена",
            "руб",
        )
        if any(keyword in lowered for keyword in domain_keywords):
            return False

        if re.search(r"\bрасскажи про\b", lowered):
            return True
        if re.search(r"\bлюбишь\b", lowered) or re.search(r"\bчто думаешь\b", lowered):
            return True
        if len(lowered.split()) <= 2:
            return True
        return False

    @staticmethod
    def _offtopic_response() -> str:
        return "Могу немного поддержать разговор, но лучше всего я разбираюсь в сантехнике. Если хотите, могу плавно вернуться к выбору товаров."

    def _get_session(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self.sessions:
            self.sessions[conversation_id] = ConversationState()
        return self.sessions[conversation_id]

    def _build_soft_promo(self, base_intent: str | None = None) -> str | None:
        matched = [scenario for scenario in self.ad_scenarios if base_intent and base_intent in scenario.get("trigger_intents", [])]
        if not matched:
            matched = self.ad_scenarios[:]
        if not matched:
            return None

        scenario = random.choice(matched)
        product = get_product_by_id(scenario["product_id"], self.products)
        if product:
            return (
                f"Кстати, если захотите посмотреть что-то из каталога, могу показать {product['name']} "
                f"за {product['price_rub']} руб. {scenario['messages'][0]}"
            )
        return f"Кстати, могу показать рекламный вариант: {scenario['product_name']}."

    def _append_plumbing_bridge(self, answer: str, state: ConversationState) -> str:
        bridge = self.RETURN_TO_PLUMBING_PHRASES[state.turns % len(self.RETURN_TO_PLUMBING_PHRASES)]
        if bridge in answer:
            return answer
        return answer.rstrip() + "\n\n" + bridge

    def _maybe_add_soft_promo(self, answer: str, state: ConversationState, intent: str) -> str:
        if state.promo_cooldown > 0:
            state.promo_cooldown -= 1
            return answer

        if intent in {"small_talk", "bot_capabilities", "ask_bot_identity"} and state.free_talk_turns >= 2:
            promo = self._build_soft_promo()
            if promo:
                state.promo_cooldown = 3
                state.free_talk_turns = 0
                return answer + "\n\n" + promo

        if intent == "request_recommendation":
            promo = self._build_soft_promo("promo_offer")
            if promo:
                state.promo_cooldown = 2
                return answer + "\n\n" + promo

        return answer

    @staticmethod
    def _rule_based_intent(text: str) -> str | None:
        lowered = text.lower().strip()

        if re.search(r"\bкак (у тебя )?дела\b", lowered) or re.search(r"\bкак ты\b", lowered):
            return "small_talk"
        if re.search(r"\bкак настроение\b", lowered) or re.search(r"\bчто нового\b", lowered):
            return "small_talk"
        if re.search(r"\bчто ты умеешь\b", lowered) or re.search(r"\bчем ты можешь помочь\b", lowered):
            return "bot_capabilities"
        if re.search(r"\bты бот\b", lowered) or re.search(r"\bты кто\b", lowered) or re.search(r"\bкто ты\b", lowered):
            return "ask_bot_identity"
        return None

    @staticmethod
    def _is_domain_intent(intent: str) -> bool:
        return intent in {
            "show_catalog",
            "select_kitchen_faucet",
            "select_bathroom_faucet",
            "select_shower_system",
            "select_toilet",
            "select_installation",
            "select_sink",
            "select_water_heater",
            "ask_price",
            "ask_characteristics",
            "selection_by_budget",
            "promo_offer",
            "compare_products",
            "select_by_room",
            "select_for_small_space",
            "select_for_renovation",
            "ask_material",
            "ask_dimensions_compatibility",
            "select_premium_upgrade",
            "request_recommendation",
            "select_complete_set",
            "ask_availability",
            "ask_delivery_installation",
        }

    def _finalize_non_target_answer(self, answer: str, state: ConversationState, intent: str = "small_talk") -> str:
        answer = self._append_plumbing_bridge(answer, state)
        return self._maybe_add_soft_promo(answer, state, intent)

    def _build_non_target_payload(
        self,
        answer: str,
        state: ConversationState,
        processed: object,
        intent: str = "small_talk",
        reply_kind: str = "small_talk",
    ) -> dict[str, str]:
        state.free_talk_turns += 1
        state.last_intent = intent
        state.last_reply_kind = reply_kind
        final_answer = self._finalize_non_target_answer(answer, state, intent)
        return {
            "intent": intent,
            "topic": processed.topic,
            "sentiment": processed.sentiment_label,
            "answer": final_answer,
        }

    def reply(self, text: str, conversation_id: str = "default") -> dict[str, str]:
        state = self._get_session(conversation_id)
        state.turns += 1
        processed = preprocess_user_text(text)

        if self.FOLLOW_UP_RECOMMENDATION_RE.match(text.strip()) and state.last_recommendation_ids:
            answer = self._get_more_recommendations(state)
            state.last_intent = "request_recommendation"
            state.last_reply_kind = "recommendation"
            return {
                "intent": "request_recommendation",
                "topic": processed.topic,
                "sentiment": processed.sentiment_label,
                "answer": answer or self._random_response("request_recommendation"),
            }

        forced_intent = self._rule_based_intent(text)
        try:
            intent = predict_intent(text)
        except Exception:
            intent = "fallback"
        if forced_intent:
            intent = forced_intent

        dialogue_answer = find_dialogue_answer(text)
        thematic_dialogue_answer = None if dialogue_answer else find_thematic_dialogue_answer(text)
        category_answer, category_items = self._handle_category_request(text)
        has_domain_markers = bool(
            processed.entities.get("product_names")
            or processed.entities.get("product_ids")
            or processed.entities.get("categories")
            or find_category_in_query(text)
            or extract_budget(text) is not None
            or processed.topic in {"catalog", "kitchen", "bathroom", "sanitary", "heating", "promo", "price"}
        )

        if (
            dialogue_answer
            and not has_domain_markers
            and not forced_intent
            and not self._is_domain_intent(intent)
        ):
            return self._build_non_target_payload(dialogue_answer, state, processed)

        if (
            thematic_dialogue_answer
            and not has_domain_markers
            and not forced_intent
            and not self._is_domain_intent(intent)
        ):
            return self._build_non_target_payload(thematic_dialogue_answer, state, processed)

        if (
            self._looks_like_offtopic(text)
            and not has_domain_markers
            and intent not in {"greeting", "goodbye", "small_talk", "bot_capabilities", "ask_bot_identity"}
        ):
            answer = dialogue_answer or thematic_dialogue_answer or self._offtopic_response()
            return self._build_non_target_payload(answer, state, processed, reply_kind="offtopic")

        if category_answer:
            answer = category_answer
            state.last_reply_kind = "recommendation"
            state.last_recommendation_ids = [item["id"] for item in category_items]
        elif intent == "show_catalog":
            answer = self._handle_show_catalog()
            state.last_reply_kind = "generic"
        elif intent in {"bot_capabilities", "ask_bot_identity", "ask_availability", "ask_delivery_installation"}:
            answer = self._random_response(intent)
            state.last_reply_kind = "generic"
        elif intent == "ask_price":
            referenced_product = self._handle_referenced_product(text, state)
            if referenced_product:
                answer = f"{referenced_product['name']} стоит {referenced_product['price_rub']} руб."
            else:
                answer = self._handle_product_price(text) or self._random_response(intent)
            state.last_reply_kind = "product"
        elif intent in {"ask_characteristics", "ask_material", "ask_dimensions_compatibility"}:
            referenced_product = self._handle_referenced_product(text, state)
            if referenced_product:
                answer = format_product_details(referenced_product)
            else:
                answer = self._handle_product_characteristics(text) or self._random_response(intent)
            state.last_reply_kind = "product"
        elif intent == "compare_products":
            answer = self._handle_compare_products(text)
            state.last_reply_kind = "product"
        elif intent == "selection_by_budget":
            answer = self._handle_budget(text, intent)
            state.last_reply_kind = "recommendation"
        elif intent == "promo_offer":
            answer = self._handle_showcase_promoted()
            state.last_reply_kind = "promo"
        elif intent == "request_recommendation":
            answer = self._handle_recommendations(text, "select_premium_upgrade") or self._random_response(intent)
            state.last_reply_kind = "recommendation"
        elif intent == "select_complete_set":
            answer = self._handle_complete_set(text)
            state.last_reply_kind = "recommendation"
        elif intent in {
            "select_kitchen_faucet",
            "select_bathroom_faucet",
            "select_shower_system",
            "select_toilet",
            "select_installation",
            "select_sink",
            "select_water_heater",
            "select_by_room",
            "select_for_small_space",
            "select_for_renovation",
            "select_premium_upgrade",
        }:
            answer = self._handle_recommendations(text, intent) or self._random_response(intent)
            promo = self._handle_promo(intent)
            if promo:
                answer = answer + "\n\n" + promo
            state.last_reply_kind = "recommendation"
        elif intent in {"greeting", "goodbye", "small_talk"}:
            answer = self._random_response(intent)
            state.last_reply_kind = "small_talk"
        else:
            answer = None
            state.last_reply_kind = "generic"

        if not answer:
            answer = dialogue_answer or thematic_dialogue_answer
        if not answer:
            answer = self._random_response("fallback")
            state.last_reply_kind = "generic"

        if intent in {"small_talk", "bot_capabilities", "ask_bot_identity"}:
            state.free_talk_turns += 1
        elif intent in {"greeting", "goodbye"}:
            state.free_talk_turns = max(0, state.free_talk_turns - 1)
        else:
            state.free_talk_turns = 0

        if state.last_reply_kind == "recommendation" and not category_items:
            recommended = recommend_products(text, self.products, intent=intent, limit=5)
            state.last_recommendation_ids = [item["id"] for item in recommended]
        elif state.last_reply_kind not in {"product", "recommendation"}:
            state.last_recommendation_ids = state.last_recommendation_ids if state.last_reply_kind == "small_talk" else None

        if intent in self.NON_TARGET_INTENTS:
            answer = self._finalize_non_target_answer(answer, state, intent)
        else:
            answer = self._maybe_add_soft_promo(answer, state, intent)
        state.last_intent = intent

        return {
            "intent": intent,
            "topic": processed.topic,
            "sentiment": processed.sentiment_label,
            "answer": answer,
        }
