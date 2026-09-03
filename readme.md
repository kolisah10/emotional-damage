# 🎭 Text Emotion Classifier

A full-stack ML project that predicts the emotion behind a sentence — trained from scratch, served through a FastAPI backend, and exposed through a Streamlit UI, deployed on Render.

**Live demo:** [https://emotional-frontend-unk9.onrender.com]
**API:** https://emotional-backend-mson.onrender.com (see `/health` and `/predict`)

> Note: both services run on Render's free tier and sleep after 15 minutes of inactivity — the first request after a period of inactivity can take 30–60 seconds to wake up.

---

## What it does

Given a sentence, the model predicts one of 6 emotions:

`sadness` · `joy` · `love` · `anger` · `fear` · `surprise`

along with a confidence score and the full probability breakdown across all 6 classes.

---

## How it works

**Model**
- Trained on the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset
- Compared plain RNN, LSTM, and GRU baselines before landing on a **Bidirectional GRU** (2-layer, 128→64 units, 300D embeddings, dropout 0.5) as the best performer
- Text preprocessing: Keras `Tokenizer` (10k word vocab, `<unk>` for out-of-vocabulary words) → padded/truncated to 50 tokens

**Backend (`/backend`)**
- FastAPI app that loads the trained model + tokenizer once at startup
- `POST /predict` — takes `{"text": "..."}`, returns the predicted emotion, confidence, and full probability breakdown
- `GET /health` — liveness check
- Full TensorFlow/Keras for inference

**Frontend (`/frontend`)**
- Streamlit app — text box, Predict button, bar chart of all 6 probabilities
- Talks to the backend over plain HTTP (configured via an `API_URL` environment variable), so the two services are fully decoupled

**Deployment**
- Two separate Render Web Services (free tier), one for the backend and one for the frontend
- Python version pinned via `PYTHON_VERSION` to stay compatible with TensorFlow's supported versions

---

## Project structure

```
.
├── backend/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt
│   └── Artifacts/
│       ├── BiGRU_Modle.keras
│       └── tokenizer.pkl
└── frontend/
    ├── streamlit_app.py     # Streamlit UI
    └── requirements.txt
```

---

## Running locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend** (in a separate terminal)
```bash
cd frontend
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

Then open the Streamlit URL it prints (usually `http://localhost:8501`).

---

## Tech stack

Python · TensorFlow/Keras · FastAPI · Streamlit · Render

---

## Notes / limitations

- The model can be less reliable on very short or generic inputs (e.g. "I am sad") compared to fuller sentences, since the training data skews toward longer, more descriptive text.
- CPU inference (both locally and on Render) can occasionally differ very slightly from GPU-trained predictions on borderline/ambiguous cases, due to floating-point differences between execution paths — not a bug, just a known characteristic of cross-hardware inference.
