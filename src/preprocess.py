import re
import pandas as pd
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Clean raw news text:
    - lowercase
    - remove punctuation & numbers
    - remove stopwords
    """
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove punctuation & numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]

    return " ".join(words)


def preprocess_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # keep only relevant columns
    df = df[["text", "label"]]

    # clean text column
    df["clean_text"] = df["text"].apply(clean_text)

    return df

if __name__ == "__main__":
    print("Starting preprocessing...")
    data = preprocess_dataset("data/news.csv")
    print("Preprocessing completed!")
    print(data.head(3))
    print("Total rows:", len(data))
