import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from detector.detector import FakeNewsDetector
import torch

# -------------------------------
# Load GPT-2 for Fake News Generation
# -------------------------------
@st.cache_resource
def load_gpt2():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

# -------------------------------
# Load Fake News Detector
# -------------------------------
@st.cache_resource
def load_detector():
    return FakeNewsDetector(model_path="./models/bert_fake_news_detector")

# Initialize models
gpt2_tokenizer, gpt2_model = load_gpt2()
detector = load_detector()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📰 Fake News Generator & Detector")

menu = ["Generate Fake News", "Detect Fake News"]
choice = st.sidebar.radio("Choose Action", menu)

if choice == "Generate Fake News":
    st.header("🛠 Fake News Headline Generator")
    prompt = st.text_input("Enter a topic (e.g., politics, technology, sports):")
    max_length = st.slider("Max Length", 30, 150, 50)

    if st.button("Generate"):
        if prompt.strip() == "":
            st.warning("Please enter a topic first!")
        else:
            inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
            outputs = gpt2_model.generate(inputs, max_length=max_length, num_return_sequences=1, do_sample=True, top_k=50)
            text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("Generated Fake News Headline:")
            st.write(text)

elif choice == "Detect Fake News":
    st.header("✅ Fake News Detector")
    user_text = st.text_area("Enter news text to check:")

    if st.button("Analyze"):
        if user_text.strip() == "":
            st.warning("Please enter some text!")
        else:
            prediction = detector.predict(user_text)
            if prediction == "Fake":
                st.error("⚠ This news seems to be **FAKE**!")
            else:
                st.success("✅ This news seems to be **REAL**.")

