import streamlit as st
from detector.detector import FakeNewsDetector
from transformers import pipeline

# Cache resources to speed up loading
@st.cache_resource
def load_detector():
    return FakeNewsDetector()

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")  # Lighter model for faster load

# Streamlit UI
st.title("📰 Fake News Generator & Detector")
st.write("Choose an option below:")

# Warning message
st.warning("⚠ This app is for demonstration only. The detection model is based on sentiment analysis and may not be accurate.")

option = st.radio("Select Feature", ["Detect Fake News", "Generate Fake News"])

if option == "Detect Fake News":
    st.subheader("Fake News Detection")
    detector = load_detector()
    user_input = st.text_area("Enter news text:")
    if st.button("Detect"):
        if user_input.strip():
            with st.spinner("Analyzing... Please wait"):
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
            with st.spinner("Generating fake news... Please wait"):
                output = generator(
    user_prompt,
    max_length=80,             # Increase token length for longer sentences
    num_return_sequences=1,    # How many outputs you want
    temperature=0.9,           # Adds randomness for creativity (higher = more random)
    top_k=50,                  # Limits sampling to top 50 tokens
    top_p=0.95,                # Nucleus sampling for diversity
    do_sample=True             # Enables sampling for variation
)

            st.success(f"Generated Fake News: {output[0]['generated_text']}")
        else:
            st.error("Please enter a topic.")




