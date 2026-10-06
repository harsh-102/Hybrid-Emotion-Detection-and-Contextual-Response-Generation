from config.config import PROMPT_PATH


def build_emotion_prompt(
    text: str,
    emotion: str,
    confidence: float
) -> str:

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    if not emotion:
        raise ValueError("Emotion cannot be empty.")

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {PROMPT_PATH}"
        )

    prompt_template = PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    prompt = prompt_template.format(
        emotion=emotion,
        confidence_percent=confidence * 100,
        text=text
    )

    return prompt.strip()