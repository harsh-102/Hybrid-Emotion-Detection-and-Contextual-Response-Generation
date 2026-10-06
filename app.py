import html

import streamlit as st

from src.emotion_predictor import predict_emotion
from src.llm_service import generate_emotion_analysis
from config.config import LOW_CONFIDENCE_THRESHOLD


st.set_page_config(
    page_title="Emotion Detection & Contextual Response",
    page_icon="💭",
    layout="centered"
)


# ======================================================================
# UI THEME (visual only, nothing here touches the model / LLM logic)
# ======================================================================

# One colour + emoji per emotion class the model predicts.
EMOTION_STYLE = {
    "joy":      ("😄", "#FFC94D"),
    "sadness":  ("😢", "#5B8DEF"),
    "anger":    ("😠", "#FF5D5D"),
    "fear":     ("😨", "#A678E8"),
    "surprise": ("😲", "#FF8FC7"),
    "love":     ("❤️", "#FF6B81"),
}
DEFAULT_STYLE = ("💭", "#8B7CFF")


def inline(markup: str) -> str:
    """Collapse indentation so Streamlit's markdown never treats HTML as a code block."""
    return "".join(line.strip() for line in markup.splitlines())


def render_html(markup: str, target=st):
    target.markdown(inline(markup), unsafe_allow_html=True)


BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');

:root { --accent: #8B7CFF; }

.stApp {
    background:
        radial-gradient(900px 480px at 50% -8%, color-mix(in srgb, var(--accent) 28%, transparent), transparent 70%),
        #0E1020;
    color: #ECEEF8;
    font-family: 'DM Sans', sans-serif;
    transition: background 0.6s ease;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 4rem; }

.hero { text-align: center; margin-bottom: 1.6rem; }
.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    line-height: 1.15;
    font-weight: 700;
    color: #F5F6FF;
    margin: 0 0 0.6rem 0;
    padding: 0;
}
.hero p {
    color: #9AA0C2;
    font-size: 1.02rem;
    line-height: 1.6;
    max-width: 560px;
    margin: 0 auto;
}

.chips { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 1.2rem 0 1.8rem 0; }
.chip {
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid #2A2E52;
    background: #151830;
    color: #9AA0C2;
    font-size: 0.88rem;
    transition: all 0.35s ease;
}
.chip.active {
    color: #0E1020;
    font-weight: 600;
    background: var(--chip);
    border-color: var(--chip);
    box-shadow: 0 0 22px color-mix(in srgb, var(--chip) 55%, transparent);
    transform: scale(1.08);
}

.stTextArea label p { color: #C9CCE6; font-weight: 500; font-size: 0.95rem; }
.stTextArea textarea {
    background: #151830 !important;
    color: #ECEEF8 !important;
    border: 1px solid #2A2E52 !important;
    border-radius: 16px !important;
    padding: 16px !important;
    font-size: 1.02rem !important;
    line-height: 1.55 !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 30%, transparent) !important;
}
.stTextArea textarea::placeholder { color: #626892 !important; }

.stButton > button {
    border-radius: 14px;
    border: 1px solid #2A2E52;
    background: #151830;
    color: #C9CCE6;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    border-color: var(--accent);
    color: #FFFFFF;
    transform: translateY(-2px);
}
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #7C6CFF, #FF7BC8);
    border: none;
    color: #FFFFFF;
    font-size: 1.05rem;
    font-weight: 600;
    padding: 0.7rem 1rem;
    box-shadow: 0 10px 30px rgba(124, 108, 255, 0.35);
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    color: #FFFFFF;
    box-shadow: 0 14px 38px rgba(255, 123, 200, 0.45);
}

.try-label { color: #7E84AB; font-size: 0.85rem; margin: 0.9rem 0 0.4rem 0; }

.result {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 20px;
    padding: 24px 26px;
    margin: 1.4rem 0 1rem 0;
    border-radius: 22px;
    background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 22%, #151830), #151830 70%);
    border: 1px solid color-mix(in srgb, var(--accent) 55%, transparent);
    box-shadow: 0 18px 50px color-mix(in srgb, var(--accent) 22%, transparent);
    animation: pop 0.5s cubic-bezier(.2,.9,.3,1.2);
}
.result-emoji { font-size: 3.6rem; line-height: 1; }
.result-label { color: #9AA0C2; font-size: 0.85rem; }
.result-name {
    font-family: 'Fraunces', serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.1;
}
.conf { flex: 1 1 220px; min-width: 200px; }
.conf-top { display: flex; justify-content: space-between; color: #C9CCE6; font-size: 0.9rem; margin-bottom: 8px; }
.conf-top b { color: #FFFFFF; }
.bar { height: 10px; border-radius: 999px; background: #0E1020; overflow: hidden; }
.fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 60%, #fff), var(--accent));
    animation: grow 1.1s ease-out;
}

.panel {
    padding: 20px 24px;
    margin: 0 0 1rem 0;
    border-radius: 18px;
    background: #151830;
    border: 1px solid #2A2E52;
}
.panel h4 {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #F5F6FF;
    margin: 0 0 10px 0;
    padding: 0;
}
.panel p { color: #C9CCE6; line-height: 1.7; margin: 0; }

.cues { display: flex; flex-wrap: wrap; gap: 8px; }
.cue {
    padding: 6px 14px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--accent) 16%, #151830);
    border: 1px solid color-mix(in srgb, var(--accent) 45%, transparent);
    color: #ECEEF8;
    font-size: 0.92rem;
}

.reply {
    padding: 22px 26px;
    margin: 0 0 1rem 0;
    border-radius: 18px;
    background: #151830;
    border-left: 4px solid var(--accent);
}
.reply h4 {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #F5F6FF;
    margin: 0 0 10px 0;
    padding: 0;
}
.reply p { color: #ECEEF8; font-size: 1.05rem; line-height: 1.75; margin: 0; }

[data-testid="stAlert"] { border-radius: 14px; }

@keyframes pop {
    from { opacity: 0; transform: translateY(14px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes grow { from { width: 0; } }

@media (prefers-reduced-motion: reduce) {
    .result, .fill { animation: none; }
    .chip, .stButton > button { transition: none; }
}
@media (max-width: 640px) {
    .hero h1 { font-size: 2rem; }
    .result-name { font-size: 1.9rem; }
}
</style>
"""

render_html(BASE_CSS)


def chips_markup(active=None) -> str:
    items = []
    for name, (emoji, color) in EMOTION_STYLE.items():
        cls = "chip active" if name == active else "chip"
        items.append(
            f'<span class="{cls}" style="--chip:{color}">{emoji} {name.capitalize()}</span>'
        )
    return '<div class="chips">' + "".join(items) + "</div>"


# ======================================================================
# HEADER
# ======================================================================

render_html(
    """
    <div class="hero">
        <h1>💭 Emotion Detection &amp; Contextual Response</h1>
        <p>Enter a sentence and the system will detect the emotion
        using a fine-tuned DistilBERT model and generate a
        contextual response using an LLM.</p>
    </div>
    """
)

# Placeholder so the row of emotions can light up after an analysis
chips_slot = st.empty()
render_html(chips_markup(), chips_slot)


# ======================================================================
# INPUT
# ======================================================================

EXAMPLES = {
    "🎉 Good news": "I finally got the internship I wanted!",
    "😤 Frustrated": "They cancelled on me again at the last minute and didn't even apologise.",
    "👻 Spooked": "Something moved in the dark hallway and I froze on the spot.",
}


def use_example(sample_text):
    st.session_state["user_text"] = sample_text


text = st.text_area(
    "Enter your text",
    placeholder="Example: I finally got the internship I wanted!",
    height=150,
    key="user_text"
)

render_html('<div class="try-label">Or try an example</div>')
example_cols = st.columns(len(EXAMPLES))
for col, (label, sample) in zip(example_cols, EXAMPLES.items()):
    col.button(
        label,
        key=f"example_{label}",
        on_click=use_example,
        args=(sample,),
        use_container_width=True
    )

st.write("")

analyze_button = st.button(
    "🔍 Analyze Emotion",
    type="primary",
    use_container_width=True
)


# ======================================================================
# PROCESSING + RESULTS (logic unchanged)
# ======================================================================

if analyze_button:

    if not text.strip():

        st.warning("Please enter some text before analyzing.")

    else:

        try:

            # Step 1: Emotion classification
            with st.spinner("Detecting emotion..."):

                prediction = predict_emotion(text)

            emotion = prediction["emotion"]
            confidence = prediction["confidence"]
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    "The model is not highly confident about this prediction. "
                    "The detected emotion should be interpreted with caution."
                )

            # Step 2: LLM analysis
            with st.spinner("Generating contextual analysis..."):

                analysis = generate_emotion_analysis(
                    text=text,
                    emotion=emotion,
                    confidence=confidence
                )


            st.success("Analysis completed!")


            # Visual theme for the detected emotion
            key = str(emotion).lower()
            emoji, color = EMOTION_STYLE.get(key, DEFAULT_STYLE)

            # Re-tint the whole page + light up the matching chip
            render_html(f"<style>:root {{ --accent: {color}; }}</style>")
            render_html(chips_markup(active=key), chips_slot)


            # -------------------------
            # Emotion Result
            # -------------------------

            render_html(
                f"""
                <div class="result">
                    <div class="result-emoji">{emoji}</div>
                    <div>
                        <div class="result-label">Detected emotion</div>
                        <div class="result-name">{html.escape(str(emotion).capitalize())}</div>
                    </div>
                    <div class="conf">
                        <div class="conf-top"><span>Confidence</span><b>{confidence * 100:.2f}%</b></div>
                        <div class="bar"><div class="fill" style="width:{confidence * 100:.2f}%"></div></div>
                    </div>
                </div>
                """
            )


            # -------------------------
            # LLM Explanation
            # -------------------------

            render_html(
                f"""
                <div class="panel">
                    <h4>Why this emotion?</h4>
                    <p>{html.escape(str(analysis["emotion_explanation"]))}</p>
                </div>
                """
            )


            # -------------------------
            # Emotional Cues
            # -------------------------

            cue_markup = "".join(
                f'<span class="cue">{html.escape(str(cue))}</span>'
                for cue in analysis["emotional_cues"]
            )

            render_html(
                f"""
                <div class="panel">
                    <h4>Emotional Cues</h4>
                    <div class="cues">{cue_markup}</div>
                </div>
                """
            )


            # -------------------------
            # Contextual Response
            # -------------------------

            render_html(
                f"""
                <div class="reply">
                    <h4>💬 Contextual Response</h4>
                    <p>{html.escape(str(analysis["contextual_response"]))}</p>
                </div>
                """
            )


        except Exception as error:

            st.error(
                "Something went wrong while processing your input."
            )

            st.caption(
                f"Error details: {error}"
            )