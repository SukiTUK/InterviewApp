"""STAR Interview Coach - Streamlit app.

Run with:  streamlit run app.py
"""
import json
import time

import pandas as pd
import streamlit as st

from guards import check_input, check_job_description, check_output, check_rate_limit
from evaluation import (CRITERIA, JUDGE_PROMPT, TEST_QUESTION, TEST_SET,
                        build_judge_message, in_range, parse_judge)
from llm import DEFAULT_SETTINGS, MODELS, chat
from pricing import calculate_cost, format_cost, get_prices
from prompts import (
    FEEDBACK_PROMPTS,
    QUESTION_GENERATOR,
    build_feedback_prompt,
    build_user_message,
    wrap_job_description,
)

st.set_page_config(page_title="STAR Interview Coach", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------- session state
defaults = {"question": "", "request_count": 0, "history": [], "total_cost": 0.0}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    role = st.selectbox(
        "Target role",
        ["Data Analyst", "Software Engineer", "Product Manager",
         "Marketing Specialist", "Project Manager", "Customer Support"],
    )
    difficulty = st.select_slider("Difficulty", ["Junior", "Mid", "Senior"], value="Junior")
    prompt_name = st.selectbox("Prompt technique", list(FEEDBACK_PROMPTS.keys()))
    st.divider()
    # ---- Model settings (Medium task 1)
    with st.expander("🛠️ Model settings (advanced)"):
        model = st.selectbox("Model", MODELS,
                             help="mini = balanced, nano = cheapest/fastest, gpt-5 = smartest/most expensive")
        temperature = st.slider("Temperature", 0.0, 2.0, DEFAULT_SETTINGS["temperature"], 0.1,
                                help="Randomness. 0 = same answer every time, higher = more creative. "
                                     "Note: GPT-5 reasoning models may ignore this.")
        max_tokens = st.slider("Max tokens", 500, 16000, DEFAULT_SETTINGS["max_tokens"], 500,
                               help="Upper limit on output length. For reasoning models this "
                                    "includes hidden 'thinking' tokens, so don't set it too low.")
        reasoning_effort = st.select_slider(
            "Reasoning effort", ["minimal", "low", "medium", "high"],
            value=DEFAULT_SETTINGS["reasoning_effort"],
            help="How long the model thinks before answering. Higher = slower, more tokens, "
                 "often better judgement.")
    settings = {"model": model, "temperature": temperature,
                "max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
    st.caption(f"Model: `{model}`")
    # ---- Price info (Medium task 3)
    prices, is_live = get_prices()
    if model in prices:
        in_p, out_p = prices[model]
        st.caption(f"Price: \\${in_p * 1e6:.2f} input / \\${out_p * 1e6:.2f} output per 1M tokens "
                   f"({'live from OpenRouter' if is_live else 'offline estimate'})")
    st.caption(f"💰 Spent this session: {format_cost(st.session_state.total_cost)}")
    st.caption(f"Requests this session: {st.session_state.request_count}")
    with st.expander("See the system prompt"):
        st.code(build_feedback_prompt(prompt_name, role, difficulty,
                                      bool(st.session_state.get("jd", "").strip())), language="text")


def call_llm(system_prompt: str, user_message: str, override: dict | None = None):
    """Rate-limited LLM call with timing. Returns (text, usage, seconds) or None.
    override: optional settings that replace the sidebar ones (e.g. a different judge model)."""
    call_settings = {**settings, **(override or {})}
    ok, msg = check_rate_limit(st.session_state.request_count)
    if not ok:
        st.error(msg)
        return None
    st.session_state.request_count += 1
    start = time.time()
    try:
        text, usage = chat(system_prompt, user_message, call_settings)
    except Exception as e:  # network / auth errors
        st.error(f"API error: {e}")
        return None
    cost = calculate_cost(call_settings["model"], usage)
    usage["cost"] = cost
    st.session_state.total_cost += cost or 0
    return text, usage, round(time.time() - start, 1)


def render_feedback(text: str, is_json_prompt: bool):
    """Show the feedback. JSON prompt gets a nicer layout."""
    if not is_json_prompt:
        st.markdown(text)
        return
    try:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        st.warning("Model did not return valid JSON. Raw output:")
        st.code(text)
        return
    if data.get("error"):
        st.error("This app only gives feedback on interview answers.")
        return
    st.metric("Score", f"{data.get('score', '?')}/10")
    icons = {"present": "✅", "weak": "⚠️", "missing": "❌"}
    cols = st.columns(4)
    for col, (part, status) in zip(cols, data.get("star", {}).items()):
        col.markdown(f"**{part.title()}**  \n{icons.get(status, '')} {status}")
    c1, c2 = st.columns(2)
    c1.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in data.get("strengths", [])))
    c2.markdown("**Improvements**\n" + "\n".join(f"- {s}" for s in data.get("improvements", [])))
    st.markdown("**Improved answer**")
    st.info(data.get("improved_answer", ""))


def get_feedback(name: str, question: str, answer: str, jd: str | None = None):
    jd = job_description if jd is None else jd
    system = build_feedback_prompt(name, role, difficulty, bool(jd.strip()))
    user = build_user_message(question, answer, role, jd)
    result = call_llm(system, user)
    if result is None:
        return None
    text, usage, seconds = result
    ok, msg = check_output(text)
    if not ok:
        st.error(msg)
        return None
    return text, usage, seconds


# ---------------------------------------------------------------- main page
st.title("🎯 STAR Interview Coach")
st.write("Practice behavioral interview questions and get feedback using the **STAR** method "
         "(Situation, Task, Action, Result).")

# ---- Optional job description (Medium task 6)
with st.expander("📄 Job description (optional) — tailor questions and feedback to a real job"):
    job_description = st.text_area(
        "Paste the job ad you are applying for",
        key="jd", height=200,
        placeholder="We are looking for a Junior Data Analyst who is skilled in SQL, Python...",
    )
    jd_ok, jd_msg = check_job_description(job_description)
    if not jd_ok:
        st.error(jd_msg)
        job_description = ""          # never send a blocked JD to the model
    elif job_description.strip():
        st.success(f"Job description added ({len(job_description)} characters). "
                   "Questions and feedback will be tailored to it.")

# ---- Step 1: question (shared by both tabs)
with st.container(border=True):
    st.subheader("1. Interview question")
    col_q, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("🎲 Generate question", width="stretch"):
            system = QUESTION_GENERATOR.replace("{role}", role).replace("{difficulty}", difficulty)
            result = call_llm(system, wrap_job_description(job_description) + "Give me a question.")
            if result:
                st.session_state.question = result[0]
    with col_q:
        st.session_state.question = st.text_input(
            "Question (generate one or type your own)",
            value=st.session_state.question,
            placeholder="Tell me about a time you solved a difficult problem.",
        )
    answer = st.text_area("2. Your answer", height=180,
                          placeholder="Situation... Task... Action... Result...")

question = st.session_state.question

tab_practice, tab_compare, tab_eval, tab_history = st.tabs(
    ["Practice", "Compare prompts", "⚖️ Evaluate (LLM judge)", "History"])

# ---- Practice tab: one prompt
with tab_practice:
    if st.button("Get feedback", type="primary"):
        ok, msg = check_input(answer)
        if not question:
            st.warning("Generate or type a question first.")
        elif not ok:
            st.error(msg)
        else:
            with st.spinner("Coach is reading your answer..."):
                result = get_feedback(prompt_name, question, answer)
            if result:
                text, usage, seconds = result
                render_feedback(text, "JSON" in prompt_name)
                st.caption(f"{prompt_name} · {seconds}s · {usage.get('total_tokens', '?')} tokens · "
                           f"{format_cost(usage.get('cost'))}")
                st.session_state.history.append(
                    {"prompt": prompt_name, "question": question, "answer": answer,
                     "feedback": text, "seconds": seconds, "tokens": usage.get("total_tokens"),
                     "cost": usage.get("cost")}
                )

# ---- Compare tab: run all 5 prompts on the same answer
with tab_compare:
    st.write("Runs the **same answer** through all 5 prompt techniques so you can see "
             "which one works best. (Uses 5 API calls.)")
    if st.button("Compare all 5 prompts"):
        ok, msg = check_input(answer)
        if not question:
            st.warning("Generate or type a question first.")
        elif not ok:
            st.error(msg)
        else:
            rows = []
            for name in FEEDBACK_PROMPTS:
                with st.spinner(f"Running {name}..."):
                    result = get_feedback(name, question, answer)
                if not result:
                    continue
                text, usage, seconds = result
                rows.append({"Technique": name, "Seconds": seconds,
                             "Input tokens": usage.get("prompt_tokens"),
                             "Output tokens": usage.get("completion_tokens"),
                             "Cost (USD)": format_cost(usage.get("cost"))})
                with st.expander(name, expanded=False):
                    render_feedback(text, "JSON" in name)
            if rows:
                st.subheader("Speed & cost")
                st.dataframe(rows, width="stretch", hide_index=True)

# ---- Evaluation tab: LLM-as-a-judge (Hard task 5)
with tab_eval:
    st.write(
        "Automatically tests all 5 prompt techniques on a **fixed test set** of 3 answers "
        "(weak, medium, strong). A second LLM — the **judge** — grades each piece of feedback "
        "on accuracy, actionability, faithfulness (no invented facts) and clarity (1–5). "
        "We also check if the coach's score lands in the expected range for that answer."
    )
    with st.expander("See the test set and judge prompt"):
        st.markdown(f"**Question:** {TEST_QUESTION}")
        for t in TEST_SET:
            st.markdown(f"**{t['label']}** (expected score {t['expected_range'][0]}–"
                        f"{t['expected_range'][1]}): {t['answer']}")
        st.code(JUDGE_PROMPT, language="text")

    c1, c2 = st.columns(2)
    judge_model = c1.selectbox("Judge model", MODELS, index=0,
                               help="Best practice: use a stronger model than the one being "
                                    "judged, so the judge doesn't just prefer its own style.")
    chosen = c2.multiselect("Test answers", [t["label"] for t in TEST_SET],
                            default=[t["label"] for t in TEST_SET])
    n_calls = len(FEEDBACK_PROMPTS) * len(chosen) * 2
    st.caption(f"This run makes {n_calls} API calls ({n_calls // 2} coach + {n_calls // 2} judge). "
               "Expect a few cents and 2–4 minutes.")

    if st.button("▶️ Run evaluation", type="primary", disabled=not chosen):
        tests = [t for t in TEST_SET if t["label"] in chosen]
        rows = []
        progress = st.progress(0.0, text="Starting...")
        total, done = len(FEEDBACK_PROMPTS) * len(tests), 0
        for name in FEEDBACK_PROMPTS:
            for t in tests:
                progress.progress(done / total, text=f"{name} · {t['label']} answer")
                coach = get_feedback(name, TEST_QUESTION, t["answer"], jd="")
                done += 1
                if not coach:
                    continue
                feedback, usage, seconds = coach
                judged = call_llm(JUDGE_PROMPT,
                                  build_judge_message(TEST_QUESTION, t["answer"], feedback),
                                  override={"model": judge_model, "reasoning_effort": "low"})
                grade = parse_judge(judged[0]) if judged else None
                if not grade:
                    st.warning(f"Judge output could not be parsed for {name} · {t['label']}.")
                    continue
                rows.append({
                    "Technique": name, "Answer": t["label"],
                    "Coach score": grade.get("coach_score"),
                    "Expected": f"{t['expected_range'][0]}–{t['expected_range'][1]}",
                    "In range": in_range(grade.get("coach_score"), t["expected_range"]),
                    **{c.title(): grade[c] for c in CRITERIA},
                    "Coach cost": usage.get("cost") or 0, "Seconds": seconds,
                    "Judge comment": grade.get("comment", ""),
                })
        progress.progress(1.0, text="Done")
        st.session_state.eval_rows = rows

    rows = st.session_state.get("eval_rows")
    if rows:
        df = pd.DataFrame(rows)
        crit_cols = [c.title() for c in CRITERIA]
        df["Quality (avg)"] = df[crit_cols].mean(axis=1)
        summary = (df.groupby("Technique")
                     .agg(**{"Quality (1-5)": ("Quality (avg)", "mean"),
                             **{c: (c, "mean") for c in crit_cols},
                             "Score in range": ("In range", "mean"),
                             "Avg cost ($)": ("Coach cost", "mean"),
                             "Avg seconds": ("Seconds", "mean")})
                     .sort_values("Quality (1-5)", ascending=False))
        summary["Score in range"] = (summary["Score in range"] * 100).round().astype(int).astype(str) + "%"
        winner = summary.index[0]
        st.success(f"🏆 Best technique on this test set: **{winner}** "
                   f"(quality {summary.iloc[0]['Quality (1-5)']:.2f} / 5)")
        st.subheader("Summary per technique")
        st.dataframe(summary.round({"Quality (1-5)": 2, **{c: 2 for c in crit_cols},
                                    "Avg cost ($)": 5, "Avg seconds": 1}),
                     width="stretch")
        st.subheader("All results")
        st.dataframe(df.drop(columns=["Quality (avg)"]), width="stretch", hide_index=True)
        st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False),
                           file_name="llm_judge_results.csv", mime="text/csv")

# ---- History tab
with tab_history:
    if not st.session_state.history:
        st.info("No feedback yet.")
    for item in reversed(st.session_state.history):
        with st.expander(f"{item['prompt']} — {item['question'][:60]}"):
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown(item["feedback"])
            st.caption(f"{item['seconds']}s · {item['tokens']} tokens · "
                       f"{format_cost(item.get('cost'))}")
