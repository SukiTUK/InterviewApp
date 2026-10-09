"""Small wrapper around OpenRouter (it uses the OpenAI-compatible API)."""
import os

import streamlit as st
from openai import OpenAI

# Models allowed for this project (Sprint 1 allow-list)
MODELS = ["openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5"]
MODEL = MODELS[0]  # default

# Default settings, used when the caller does not pass any
DEFAULT_SETTINGS = {
    "model": MODEL,
    "temperature": 1.0,
    "max_tokens": 4000,
    "reasoning_effort": "low",
}


def _get_api_key() -> str:
    # 1) .streamlit/secrets.toml  2) environment variable
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.getenv("OPENROUTER_API_KEY", "")


@st.cache_resource
def get_client() -> OpenAI:
    key = _get_api_key()
    if not key:
        st.error("No API key found. Add OPENROUTER_API_KEY to .streamlit/secrets.toml")
        st.stop()
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def chat(system_prompt: str, user_message: str, settings: dict | None = None) -> tuple[str, dict]:
    """Send one system + user message. Returns (text, token usage).

    settings keys:
      model            - which LLM to use
      temperature      - randomness (0 = predictable, 2 = very random)
      max_tokens       - upper limit on output length (includes hidden reasoning tokens!)
      reasoning_effort - how long a reasoning model "thinks" before answering
    """
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    client = get_client()
    resp = client.chat.completions.create(
        model=s["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=s["temperature"],
        max_tokens=s["max_tokens"],
        # 'reasoning' is an OpenRouter-specific field, so it goes in extra_body
        extra_body={"reasoning": {"effort": s["reasoning_effort"]}},
    )
    text = resp.choices[0].message.content or ""
    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    if not text.strip():
        text = ("⚠️ The model returned an empty answer. It probably used all of `max_tokens` "
                "on hidden reasoning. Increase **Max tokens** or lower **Reasoning effort**.")
    return text.strip(), usage
