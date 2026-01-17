📌 Project Overview

This project implements an end-to-end Fake News Detection system using Machine Learning and Natural Language Processing (NLP) techniques.
The system classifies news articles as FAKE or REAL based on textual patterns learned from labeled data.

The objective of this project is to demonstrate the complete ML pipeline—from preprocessing raw text to training, evaluating, and using a trained model for prediction—rather than performing factual or real-world fact verification.

🔍 Problem Statement

The rapid spread of information on digital platforms has increased the risk of misinformation. Automated systems can assist in flagging potentially misleading content by learning patterns commonly associated with fake or real news articles.

This project explores how supervised machine learning can be applied to the fake news detection problem using textual data.

🧠 Methodology & Approach

The project follows a standard supervised text-classification workflow:

1️⃣ Dataset Selection

A publicly available Fake or Real News dataset sourced from Kaggle is used.

Dataset details:

Labeled news articles classified as FAKE (0) or REAL (1)

Primary features:

text: full news article content

label: target variable

Includes optional metadata such as title or index

The dataset is predominantly English-language and Western/U.S.-centric

The dataset is used strictly for academic and learning purposes.

2️⃣ Text Preprocessing

Raw text data is cleaned using NLP techniques to make it suitable for machine learning:

Convert text to lowercase

Remove punctuation and numerical characters

Remove extra whitespaces

Remove common English stopwords

Preprocessing improves signal quality and reduces noise without altering original data.

3️⃣ Feature Extraction

TF-IDF (Term Frequency – Inverse Document Frequency) is used to convert cleaned text into numerical feature vectors.

TF-IDF captures the importance of words based on their frequency relative to the corpus.

4️⃣ Model Training

Logistic Regression is used as the classification algorithm.

Dataset split:

80% training

20% testing

The trained model achieves approximately 91–92% accuracy on the test set.

5️⃣ Prediction

The trained model and vectorizer are saved and reused.

New news text can be entered via a command-line interface.

The system outputs a prediction: FAKE or REAL.

📂 Project Structure
fake-news-detection-ml/
├── data/
│   └── news.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore

▶️ How to Run the Project
1️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python src/train.py

4️⃣ Predict Fake or Real News
python src/predict.py


Enter any news text when prompted to receive a prediction.

📊 Model Performance

Algorithm: Logistic Regression

Feature Extraction: TF-IDF

Accuracy: ~91–92%

Model accuracy reflects performance on the provided dataset and does not guarantee real-world accuracy.

⚠️ Important Note on Predictions

The system may classify apparently legitimate or neutral news as FAKE.
This behavior is expected and highlights the limitations of pattern-based machine learning approaches.

The model:

❌ Does NOT verify real-world facts

❌ Does NOT validate sources

❌ Does NOT understand context beyond learned patterns

✅ Identifies statistical similarities between text and training data

🚧 Limitations

The model is trained on a specific public dataset and learns linguistic patterns rather than factual correctness.

It does not perform real-time fact-checking or source verification.

Predictions may be biased toward the writing style and geopolitical context of the training data.

Region-specific or generic news (e.g., Indian policy or local news) may be misclassified.

The system should be used as an assistive ML tool, not as a definitive fake-news verifier.

🔮 Future Work

Incorporate Indian and region-specific news datasets to reduce domain bias.

Experiment with deep learning models such as LSTMs or Transformers.

Add source credibility analysis.

Deploy the system as a web application or API.

📚 Technologies Used

Python

Pandas

NLTK

Scikit-learn

TF-IDF

Logistic Regression

