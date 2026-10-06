from src.emotion_predictor import predict_emotion


def test_emotion_prediction_returns_required_fields():

    result = predict_emotion(
        "I am extremely happy about my success!"
    )

    assert "emotion" in result
    assert "confidence" in result

    assert isinstance(result["emotion"], str)
    assert isinstance(result["confidence"], float)

    assert 0 <= result["confidence"] <= 1