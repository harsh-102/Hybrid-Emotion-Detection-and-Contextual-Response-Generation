from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "emotion-distilbert"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "emotion_analysis.txt"


# Emotion model configuration
MAX_INPUT_LENGTH = 128


# Groq configuration
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TEMPERATURE = 0.4
GROQ_MAX_COMPLETION_TOKENS = 500
LOW_CONFIDENCE_THRESHOLD = 0.60


# Application configuration
APP_TITLE = "Emotion Detection & Contextual Response"