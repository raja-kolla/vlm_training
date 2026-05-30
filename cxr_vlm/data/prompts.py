"""Prompt and response templates for CXR report generation."""

SYSTEM_INSTRUCTION = (
    "You are an expert radiologist. Given a chest X-ray image and optional "
    "clinical history, write a structured radiology report."
)

USER_PROMPT_TEMPLATE = """<image>
{system_instruction}

Clinical history: {history}

Generate a structured chest X-ray report using exactly this format:

<observations>
(detailed imaging findings)
</observations>
<conclusion>
(brief summary / impression)
</conclusion>"""

ASSISTANT_RESPONSE_TEMPLATE = """<observations>
{observations}
</observations>
<conclusion>
{conclusion}
</conclusion>"""

DEFAULT_HISTORY = "Not provided."


def normalize_history(history: str | None) -> str:
    if history is None:
        return DEFAULT_HISTORY
    try:
        import math

        if isinstance(history, float) and math.isnan(history):
            return DEFAULT_HISTORY
    except (TypeError, ValueError):
        pass
    text = str(history).strip()
    if not text or text.lower() == "nan":
        return DEFAULT_HISTORY
    return text


def build_user_prompt(history: str | None) -> str:
    return USER_PROMPT_TEMPLATE.format(
        system_instruction=SYSTEM_INSTRUCTION,
        history=normalize_history(history),
    )


def build_assistant_response(observations: str, conclusion: str) -> str:
    return ASSISTANT_RESPONSE_TEMPLATE.format(
        observations=str(observations).strip(),
        conclusion=str(conclusion).strip(),
    )
