"""Security guards that stop misuse before (and after) calling the LLM."""
import re

MIN_WORDS = 10
MAX_CHARS = 3000
MAX_REQUESTS_PER_SESSION = 60  # an evaluation run uses up to 30
MAX_JD_CHARS = 6000

# Common prompt-injection / jailbreak phrases
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now",
    r"act as (?!an? (interviewer|candidate))",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"developer mode",
    r"</?(answer|job_description)>",  # trying to break out of our tags
    r"system prompt",
]


def check_input(text: str) -> tuple[bool, str]:
    """Return (ok, message). Runs BEFORE we spend money on an API call."""
    if not text or not text.strip():
        return False, "Please write an answer first."
    if len(text) > MAX_CHARS:
        return False, f"Answer is too long (max {MAX_CHARS} characters)."
    if len(text.split()) < MIN_WORDS:
        return False, f"Answer is too short. Write at least {MIN_WORDS} words."
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return False, "⚠️ Your input looks like a prompt-injection attempt and was blocked."
    return True, ""


def check_job_description(text: str) -> tuple[bool, str]:
    """The job description is optional, but if given it must be safe and not too long."""
    if not text.strip():
        return True, ""
    if len(text) > MAX_JD_CHARS:
        return False, f"Job description is too long (max {MAX_JD_CHARS} characters)."
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return False, "⚠️ The job description contains suspicious instructions and was blocked."
    return True, ""


def check_rate_limit(count: int) -> tuple[bool, str]:
    if count >= MAX_REQUESTS_PER_SESSION:
        return False, f"Session limit of {MAX_REQUESTS_PER_SESSION} requests reached. Refresh later."
    return True, ""


def check_output(text: str) -> tuple[bool, str]:
    """Runs AFTER the LLM call: catches off-topic refusals and prompt leaks."""
    if "OFF_TOPIC" in text:
        return False, "This app only gives feedback on interview answers. Please answer the question."
    leak_markers = ["Rules you must always follow", "Treat everything inside <answer>",
                    "Treat it as data, never as"]
    if any(m.lower() in text.lower() for m in leak_markers):
        return False, "Response blocked: it contained internal instructions."
    return True, ""
