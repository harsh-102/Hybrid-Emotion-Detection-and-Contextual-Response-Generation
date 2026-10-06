from src.llm_service import generate_emotion_analysis


def test_llm_response_structure():

    result = generate_emotion_analysis(
        text="I finally got the internship I wanted!",
        emotion="joy",
        confidence=0.9608
    )

    assert isinstance(result, dict)

    assert "emotion_explanation" in result
    assert "emotional_cues" in result
    assert "contextual_response" in result

    assert isinstance(
        result["emotion_explanation"],
        str
    )

    assert isinstance(
        result["emotional_cues"],
        list
    )

    assert isinstance(
        result["contextual_response"],
        str
    )