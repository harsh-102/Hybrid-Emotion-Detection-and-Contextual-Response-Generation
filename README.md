# 💭 EmotionLens
## Emotion Detection and Contextual Response Generation using NLP and LLMs

EmotionLens is an NLP-based application that detects emotions from user-provided text using a fine-tuned DistilBERT model and then generates an explanation, identifies emotional cues, and produces a contextual response using an LLM through the Groq API.

The project combines a supervised Transformer-based emotion classifier with an LLM to create an end-to-end emotion understanding and response-generation system.

---

## 📌 Project Overview

Traditional emotion classification systems usually stop after predicting an emotion such as joy, sadness, anger, or fear.

EmotionLens extends this process by combining:

- **DistilBERT** for emotion classification
- **Confidence estimation** for the classifier prediction
- **Prompt engineering** for controlled LLM interaction
- **Groq API** for contextual language generation
- **Structured JSON output** for consistent results
- **Streamlit** for the user interface

The system separates the responsibilities of the two models:

> **DistilBERT classifies the emotion, while the LLM explains the classification and generates a contextual response.**

This prevents the LLM from overriding the actual classifier prediction.

---

# 🎯 Objectives

The main objectives of the project are:

1. Detect the emotional category expressed in a user's text.
2. Calculate the confidence of the emotion classifier.
3. Identify words or phrases that act as emotional cues.
4. Generate a concise explanation of the detected emotion.
5. Generate a natural contextual response.
6. Integrate an LLM through an API.
7. Use prompt engineering to control LLM behavior.
8. Provide a user-friendly interactive interface.
9. Handle uncertain predictions using confidence-based warnings.
10. Maintain a clear separation between classification and language generation.

---

# 🧠 Supported Emotions

The trained classifier recognizes six emotion classes:

| Label | Emotion |
|---|---|
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

The model therefore performs a six-class text classification task.

---

# 🏗️ System Architecture

The overall architecture of EmotionLens is:

```text
                    User Input
                        │
                        ▼
              ┌───────────────────┐
              │   Streamlit UI    │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  Input Validation │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │     DistilBERT    │
              │ Emotion Classifier│
              └─────────┬─────────┘
                        │
                 Emotion + Confidence
                        │
                        ▼
              ┌───────────────────┐
              │  Prompt Builder   │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │    Groq LLM API   │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Structured Output │
              │      JSON         │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │    Streamlit UI   │
              │                   │
              │ • Emotion         │
              │ • Confidence      │
              │ • Explanation     │
              │ • Emotional Cues  │
              │ • Response        │
              └───────────────────┘


📊 Dataset
The project uses the dair-ai/emotion dataset for six-class emotion classification.
The dataset contains:
- 16,000 training examples
- 2,000 validation examples
- 2,000 test examples
The classes are:
sadness
joy
love
anger
fear
surprise

The dataset is not perfectly balanced. In particular, the surprise class contains substantially fewer examples than classes such as joy and sadness.
This class imbalance is important when interpreting model performance, which is why macro F1-score is considered alongside accuracy.


🤖 Emotion Classification Model
Fine-Tuned DistilBERT
A pretrained distilbert-base-uncased Transformer model was fine-tuned for the six-class emotion classification task.
Configuration
- Model: distilbert-base-uncased
- Maximum sequence length: 128 tokens
- Number of classes: 6
- Training epochs: 3
- Framework: PyTorch + Hugging Face Transformers
The trained model is stored locally in:
models/emotion-distilbert/

The directory contains the trained model weights, tokenizer files, model configuration, and training configuration.


📈 Model Performance
The fine-tuned DistilBERT model achieved the following results.
Validation Performance
Metric	Score
Accuracy	94.30%
Macro F1	91.97%
Weighted F1	94.29%


Test Performance
Metric	Score
Accuracy	92.55%
Macro F1	87.54%
Weighted F1	92.53%


The difference between accuracy and macro F1 is partly related to the uneven distribution of emotion classes.
The per-class test F1 scores were:
Emotion	F1 Score
Sadness	0.97
Joy	0.95
Love	0.83
Anger	0.92
Fear	0.89
Surprise	0.70


The surprise class was more difficult for the classifier than the other classes. This is consistent with its smaller representation in the training data and its overlap with other emotional expressions.


🔍 Confidence-Based Prediction
In addition to predicting an emotion, the classifier calculates a confidence score using the softmax probabilities produced by the model.
For example:
Emotion: JOY
Confidence: 96.08%

The application also handles low-confidence predictions.
The current threshold is:
60%

If the classifier confidence falls below this threshold, the application displays a warning indicating that the prediction should be interpreted with caution.
This is particularly useful for ambiguous sentences where multiple emotional interpretations may be possible.


🧠 Role of the LLM
The LLM is not responsible for classifying the emotion.
Instead, the project uses the LLM for three tasks:
1. Emotion Explanation
The LLM explains why the text reflects the emotion predicted by DistilBERT.
2. Emotional Cue Extraction
The LLM identifies words or phrases from the user's input that support the detected emotion.
3. Contextual Response Generation
The LLM generates a short and natural response appropriate to the user's emotional context.
Therefore:
DistilBERT
    ↓
Emotion Classification
    ↓
Emotion + Confidence
    ↓
LLM
    ↓
Explanation + Cues + Contextual Response

This separation makes the system more controlled and easier to evaluate.


✍️ Prompt Engineering
The project uses a dedicated prompt template stored in:
prompts/emotion_analysis.txt

The prompt receives:
- User text
- Detected emotion
- Classifier confidence
The LLM is explicitly instructed not to override the classifier's detected emotion.
The prompt also instructs the model to:
- Explain the detected emotion.
- Use evidence from the user's actual text.
- Identify emotional cues.
- Generate a concise contextual response.
- Avoid medical or psychological diagnosis.
- Avoid claiming certainty when classifier confidence is low.
- Return structured JSON.
This allows the LLM to perform controlled generation rather than unrestricted text generation.


📦 Structured LLM Output
The LLM is instructed to return a structured response containing three fields:
{
  "emotion_explanation": "Brief explanation of the detected emotion.",
  "emotional_cues": [
    "cue 1",
    "cue 2"
  ],
  "contextual_response": "A short contextual response."
}

This structured format makes it easier for the Streamlit application to display each component separately.
⚙️ Technology Stack
Technology	Purpose
Python	Core programming language
PyTorch	Model inference and training
Hugging Face Transformers	DistilBERT model
Hugging Face Tokenizers	Text tokenization
Groq API	LLM inference
Streamlit	Web interface
python-dotenv	Environment variable management
pytest	Automated testing


📁 Project Structure
emotion-detection-llm/
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── models/
│   └── emotion-distilbert/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── training_args.bin
│
├── notebooks/
│
├── prompts/
│   └── emotion_analysis.txt
│
├── src/
│   ├── __init__.py
│   ├── emotion_predictor.py
│   ├── llm_service.py
│   └── prompt_builder.py
│
├── tests/
│   ├── test_emotion_predictor.py
│   ├── test_prompt_builder.py
│   └── test_llm_service.py
│
├── .env
├── .env.example
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── test_llm.py

🔧 Description of Important Files
app.py
Main Streamlit application.
Responsible for:
- User input
- Input validation
- Calling the emotion classifier
- Calling the LLM
- Displaying the results
- Confidence warnings
- Error handling
src/emotion_predictor.py
Loads the fine-tuned DistilBERT model and performs emotion prediction.
Returns:
{    "emotion": "...",    "confidence": ...}


src/prompt_builder.py
Builds the prompt sent to the LLM using the user's text, detected emotion, and classifier confidence.
src/llm_service.py
Handles communication with the Groq API and parses the structured JSON response.
prompts/emotion_analysis.txt
Contains the LLM prompt template and instructions.
config/config.py
Contains project configuration such as:
- Model paths
- LLM configuration
- Input limits
- Confidence threshold
test_llm.py
A simple end-to-end test for:
DistilBERT → Groq → Structured Response

tests/
Contains automated tests for the main components of the application.


🚀 Installation
1. Clone or download the project
git clone <repository-url>
cd emotion-detection-llm

2. Create a virtual environment
python -m venv venv

3. Activate the environment
Windows
venv\Scripts\activate

Linux/macOS
source venv/bin/activate

4. Install dependencies
pip install -r requirements.txt


🔑 API Configuration
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key

The API key should not be hard-coded into Python source files.
A .env.example file is provided as a template:
GROQ_API_KEY=


▶️ Running the Application
Start the Streamlit application:
streamlit run app.py

The application will open in a browser.
Enter a sentence and click:


🔍 Analyze Emotion

The system will:
1. Validate the input.
2. Detect the emotion using DistilBERT.
3. Calculate classifier confidence.
4. Build an LLM prompt.
5. Send the prompt to the Groq API.
6. Receive structured JSON.
7. Display the explanation, emotional cues, and contextual response.


🧪 Running Tests
Run the automated tests with:
pytest

The test suite checks the main project components, including:
- Emotion prediction output
- Prompt generation
- LLM response structure
An additional end-to-end test can be run using:
python test_llm.py


💡 Example
Input
I finally got the internship I wanted!

Classifier Output
Emotion: JOY
Confidence: 96.08%

LLM Output
Emotion Explanation:
The user expresses excitement and satisfaction after achieving a desired goal.

Emotional Cues:
- finally
- got
- internship
- wanted

Contextual Response:
Congratulations! That's fantastic news.


⚠️ Handling Uncertain Predictions
Not every sentence clearly expresses one emotion.
For example:
Wow, I never expected this to happen.

may contain strong surprise-related language while not clearly indicating whether the event was positive or negative.
If the classifier produces a low-confidence prediction, EmotionLens displays a warning rather than presenting the prediction as completely certain.
This allows the user to interpret ambiguous predictions appropriately.


🛡️ Error Handling
The application handles several common situations:
- Empty input
- Excessively long input
- Invalid API authentication
- API rate limiting
- Temporary API unavailability
- Invalid structured LLM output
- Low-confidence emotion predictions
User-facing errors are displayed without exposing unnecessary internal implementation details.


🔒 Privacy and Security
The project does not hard-code the Groq API key.
The API key is loaded through an environment variable:
GROQ_API_KEY

The .env file should remain local and should not be committed to a public repository.


⚠️ Limitations
The project has several limitations:
1. Emotion classification depends on the training data and may produce incorrect predictions for ambiguous text.
2. The surprise class has fewer training examples and shows lower test performance than some other classes.
3. The LLM response depends on the availability and behavior of the external Groq API.
4. The system detects the emotion expressed in text but does not perform psychological or medical diagnosis.
5. Short or context-dependent sentences may be difficult to classify accurately.
6. The contextual response generated by the LLM is not intended to replace professional mental-health support.


🔮 Future Scope
Possible future improvements include:
- Expanding the emotion dataset.
- Improving performance on minority emotion classes.
- Experimenting with additional Transformer architectures.
- Adding conversation history for multi-turn contextual responses.
- Supporting multilingual emotion detection.
- Adding emotion probability visualization for all six classes.
- Adding voice input and speech-based emotion analysis.
- Deploying the application as a web service.
- Adding more extensive automated and API-mocked tests.


👩‍💻 Project Summary
EmotionLens demonstrates how a traditional NLP classification model and a modern LLM can be combined into a single application.
The project uses:
Fine-Tuned DistilBERT
        +
Confidence Estimation
        +
Prompt Engineering
        +
Groq LLM API
        +
Structured JSON
        +
Streamlit

The final system does more than simply classify an emotion: it provides an explanation of the prediction, identifies supporting emotional cues, and generates a contextual response while keeping emotion classification and language generation as separate responsibilities.