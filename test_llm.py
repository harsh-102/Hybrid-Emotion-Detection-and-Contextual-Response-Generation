from src.emotion_predictor import predict_emotion
from src.llm_service import generate_emotion_analysis


text = "I finally got the internship I wanted!"

prediction = predict_emotion(text)

print("\nCLASSIFIER RESULT")
print("-----------------")
print("Emotion:", prediction["emotion"])
print(f"Confidence: {prediction['confidence'] * 100:.2f}%")


analysis = generate_emotion_analysis(
    text=text,
    emotion=prediction["emotion"],
    confidence=prediction["confidence"]
)


print("\nLLM RESULT")
print("----------")
print(analysis)