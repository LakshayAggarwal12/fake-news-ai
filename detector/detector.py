from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FakeNewsDetector:
    def __init__(self, model_path="./models/bert_fake_news_detector"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return "Fake" if prediction == 0 else "Real"  # Adjust based on your interpretation

if __name__ == "__main__":
    detector = FakeNewsDetector()
    print(detector.predict("Breaking: Aliens land on the White House lawn!"))

