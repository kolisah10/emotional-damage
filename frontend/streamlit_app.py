import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Emotion Classifier", page_icon="🎭")

st.title("🎭 Text Emotion Classifier")
st.write("Enter a sentence and the model will predict the underlying emotion.")

text = st.text_area("Your text", placeholder="I can't believe how happy I am right now!", height=100)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Predicting..."):
                response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
            if response.status_code == 200:
                result = response.json()
                emotion = result["predicted_emotion"]
                emoji = result["emoji"]
                confidence = result["confidence"]

                st.success(f"**{emoji} Predicted emotion: {emotion.capitalize()}** ({confidence*100:.1f}% confidence)")

                st.subheader("All probabilities")
                probs = result["all_probabilities"]
                sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
                st.bar_chart(sorted_probs)
            else:
                st.error(f"API returned an error ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Couldn't reach the API at {API_URL}. Is it running?")
        except requests.exceptions.Timeout:
            st.error("The API took too long to respond. It might be waking up from sleep (Render free tier) — try again in a moment.")

st.caption(f"Connected to API: {API_URL}")
