import pickle
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- 1. Constants ----
MODEL_PATH = "Artifacts/BiGRU_Modle.keras"   # matches your actual saved filename (note: "Modle")
TOKENIZER_PATH = "Artifacts/tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 50
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness": "😔",
    "joy": "😂",
    "love": "😍",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
}


# ---- 2. Request / response schemas ----
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I am feeling great today!"},
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---- 4. Model loading + lifespan ----
ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and tokenizer...")
    ml_state["model"] = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        ml_state["tokenizer"] = pickle.load(f)
    print("Model and tokenizer loaded successfully.")

    yield

    ml_state.clear()


# ---- 5. App + CORS (only ONE FastAPI() call — this was overwritten before) ----
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a portfolio project; tighten if this becomes a real product
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 6. Endpoints ----
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="server is running", model_loaded=bool(ml_state))


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    model = ml_state.get("model")
    tokenizer = ml_state.get("tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    tokenized_text = tokenizer.texts_to_sequences([text_input.text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(padded_sequence, verbose=0)[0]
    top_emotion_index = int(np.argmax(probabilities))
    top_emotion = EMOTION_LABELS[top_emotion_index]

    all_probabilities = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probabilities)}

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=top_emotion,
        emoji=EMOTION_EMOJIS[top_emotion],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities,
    )
