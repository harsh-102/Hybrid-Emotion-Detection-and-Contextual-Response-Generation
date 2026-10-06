from src.prompt_builder import build_emotion_prompt


def test_prompt_contains_input_information():

    prompt = build_emotion_prompt(
        text="I finally got the internship I wanted!",
        emotion="joy",
        confidence=0.9608
    )

    assert "I finally got the internship I wanted!" in prompt
    assert "joy" in prompt
    assert "96.08%" in prompt