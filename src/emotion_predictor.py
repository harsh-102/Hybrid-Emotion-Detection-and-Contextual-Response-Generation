from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.config import MODEL_PATH, MAX_INPUT_LENGTH


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load model and tokenizer
# --------------------------------------------------

print(f"Loading emotion model from: {MODEL_PATH}")
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH)
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH)
)

model.to(DEVICE)
model.eval()

print("Emotion model loaded successfully.")


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_emotion(text: str) -> dict:
    """
    Predict the emotion of the given text.

    Returns:
        {
            "emotion": str,
            "confidence": float
        }
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input text cannot be empty.")

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_INPUT_LENGTH
    )

    # Move tensors to the same device as model
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    # Highest probability
    confidence, predicted_id = torch.max(
        probabilities,
        dim=-1
    )

    predicted_id = predicted_id.item()
    confidence = confidence.item()

    # Get emotion name from model configuration
    emotion = model.config.id2label[predicted_id]

    return {
        "emotion": emotion,
        "confidence": confidence
    }


# --------------------------------------------------
# Local testing
# --------------------------------------------------

if __name__ == "__main__":

    test_sentences = [
        "I finally got the internship I wanted!",
        "I am really scared about what will happen tomorrow.",
        "I am so angry that they treated me this way.",
        "I miss my family so much.",
        "I love spending time with my best friend.",
        "Wow, I did not expect that at all!"
    ]

    print("\n" + "=" * 60)
    print("EMOTION MODEL TEST")
    print("=" * 60)

    for text in test_sentences:

        result = predict_emotion(text)

        print(f"\nText: {text}")
        print(f"Emotion: {result['emotion']}")
        print(
            f"Confidence: {result['confidence'] * 100:.2f}%"
        )