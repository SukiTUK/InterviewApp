"""Price calculation (Medium task 3).

OpenRouter publishes live prices at https://openrouter.ai/api/v1/models.
Each model has a "pricing" block with the price PER TOKEN in US dollars, e.g.
    "pricing": {"prompt": "0.00000025", "completion": "0.000002", ...}

cost = prompt_tokens * prompt_price + completion_tokens * completion_price
"""
import requests
import streamlit as st

MODELS_URL = "https://openrouter.ai/api/v1/models"

# Used only if the live endpoint can't be reached (USD per 1M tokens: input, output)
FALLBACK_PER_MILLION = {
    "openai/gpt-5-mini": (0.25, 2.00),
    "openai/gpt-5-nano": (0.05, 0.40),
    "openai/gpt-5": (1.25, 10.00),
}


@st.cache_data(ttl=3600)  # fetch at most once per hour
def get_prices() -> tuple[dict, bool]:
    """Returns ({model_id: (input_price_per_token, output_price_per_token)}, is_live)."""
    try:
        data = requests.get(MODELS_URL, timeout=10).json()["data"]
        prices = {
            m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
            for m in data
            if "pricing" in m
        }
        return prices, True
    except Exception:
        fallback = {k: (i / 1_000_000, o / 1_000_000) for k, (i, o) in FALLBACK_PER_MILLION.items()}
        return fallback, False


def calculate_cost(model: str, usage: dict) -> float | None:
    """Cost of one call in USD. Reasoning tokens are billed as completion tokens."""
    prices, _ = get_prices()
    if model not in prices or not usage:
        return None
    input_price, output_price = prices[model]
    return (usage.get("prompt_tokens", 0) * input_price
            + usage.get("completion_tokens", 0) * output_price)


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "n/a"
    return f"${cost:.5f}"
