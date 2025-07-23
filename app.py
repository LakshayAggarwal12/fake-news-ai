import streamlit as st
from detector.detector import FakeNewsDetector
from transformers import pipeline

# Cache the models to avoid reloading
@st.cache_resource
def load_detector():
    return FakeNewsDetector()

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

# Streamlit UI
st.title("📰 Fake News Generator & Detector")
st.write("Choose an option below:")

option = st.radio("Select Feature", ["Detect Fake News", "Generate Fake News"])

if option == "Detect Fake News":
    st.subheader("Fake News Detection")
    detector = load_detector()
    user_input = st.text_area("Enter news text:")
    if st.button("Detect"):
        if user_input.strip():
            result = detector.predict(user_input)
            st.success(f"Prediction: {result}")
        else:
            st.error("Please enter some text.")

elif option == "Generate Fake News":
    st.subheader("Fake News Generator")
    generator = load_generator()
    user_prompt = st.text_input("Enter a topic (e.g., 'Breaking News:'):")
    if st.button("Generate"):
        if user_prompt.strip():
            output = generator(user_prompt, max_length=40, num_return_sequences=1)
            st.success(f"Generated Fake News: {output[0]['generated_text']}")
        else:
            st.error("Please enter a topic.")



