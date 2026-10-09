# 🎯 STAR Interview Coach

A Streamlit app for practicing behavioral interview questions. It gives feedback using the
STAR method (Situation, Task, Action, Result).

- **Model:** `openai/gpt-5-mini` via OpenRouter
- **5 prompt techniques:** Zero-shot, Few-shot, Chain-of-Thought, Persona, Structured JSON
- **Security guards:** input validation, prompt-injection filter, answer tagging,
  off-topic refusal, system-prompt leak check, per-session rate limit

## Setup (VS Code terminal)

```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key inside
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `prompts.py` | The 5 system prompts + question generator |
| `guards.py` | Security checks |
| `llm.py` | OpenRouter API call |

## Prompt comparison (fill this in after testing)

| Technique | Quality (1-5) | Consistency | Avg tokens | Notes |
|---|---|---|---|---|
| Zero-shot | | | | |
| Few-shot | | | | |
| Chain-of-Thought | | | | |
| Persona | | | | |
| Structured JSON | | | | |

**Winner:** ...
