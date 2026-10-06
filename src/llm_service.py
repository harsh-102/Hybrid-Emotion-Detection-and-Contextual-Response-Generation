import json
import os

from dotenv import load_dotenv
from groq import Groq

from src.prompt_builder import build_emotion_prompt


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY was not found in the .env file.")


client = Groq(api_key=GROQ_API_KEY)

from config.config import (
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    GROQ_MAX_COMPLETION_TOKENS
)

def generate_emotion_analysis(
    text: str,
    emotion: str,
    confidence: float
) -> dict:

    prompt = build_emotion_prompt(
        text=text,
        emotion=emotion,
        confidence=confidence
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an emotion analysis assistant. "
                    "Follow the user's instructions exactly."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=GROQ_TEMPERATURE,

        max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "emotion_analysis",
                "strict": True,
                "schema": {
                    "type": "object",

                    "properties": {
                        "emotion_explanation": {
                            "type": "string"
                        },

                        "emotional_cues": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },

                        "contextual_response": {
                            "type": "string"
                        }
                    },

                    "required": [
                        "emotion_explanation",
                        "emotional_cues",
                        "contextual_response"
                    ],

                    "additionalProperties": False
                }
            }
        }
    )

    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError("Groq returned an empty response.")

    return json.loads(response_text)