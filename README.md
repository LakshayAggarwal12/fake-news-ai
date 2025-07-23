# 📰 Fake News Generator & Detector using Generative AI & NLP

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-green)
![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✅ Overview
This project demonstrates how **Generative AI** can both **create and combat misinformation**.  
It includes:
- A **Fake News Generator** using **GPT-2**.
- A **Fake News Detector** using **DistilBERT** (fine-tuned for text classification).

The solution is deployed as an **interactive web app** using **Streamlit**.

---

## ✅ Features
✔ Generate fake news headlines with GPT-2  
✔ Detect if a news headline/text is **Fake** or **Real** using DistilBERT  
✔ User-friendly **Streamlit web interface**  
✔ Automatic **model download** if not available locally  

---

## ✅ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Transformers, Torch, Streamlit  
- **Models:** GPT-2 (for generation), DistilBERT (for detection)  
- **Deployment:** Streamlit Cloud / Hugging Face Spaces  

---

## ✅ Project Structure
fake-news-ai/
│
├── app.py # Main Streamlit app
├── detector/
│ └── detector.py # Fake news detection logic
├── generator/
│ └── generator.py # Fake news headline generator
├── requirements.txt # Dependencies
├── README.md # Project Documentation
└── .gitignore # Ignore models and venv

---

## ✅ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-ai.git
cd fake-news-ai
2️⃣ Create Virtual Environment & Activate
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate # On Mac/Linux
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Streamlit App
streamlit run app.py
The app will open on http://localhost:8501.

✅ Deployment
You can deploy this app for free using:

Streamlit Cloud

Hugging Face Spaces

✅ Future Scope
✔ Multi-language fake news detection
✔ Integration with real-time news APIs
✔ Advanced NLP pipelines for fact-checking

✅ License
This project is licensed under the MIT License.

