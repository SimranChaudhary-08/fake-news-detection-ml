import joblib
import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_news(text: str):
    # Load trained model and vectorizer
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    # Clean and transform text
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vectorized_text)[0]

    return prediction


if __name__ == "__main__":
    print("Fake News Detection System")
    print("--------------------------")

    user_input = input("Enter news text:\n")
    result = predict_news(user_input)

    if result == 1:
        print("\nPrediction: REAL news")
    else:
        print("\nPrediction: FAKE news")
