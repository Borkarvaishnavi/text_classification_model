import pandas as pd

df = pd.read_csv("data.csv")
texts = df["description"].fillna("")
labels = df["fraudulent"]

df["text"] = (
    df["title"].fillna("") + " " +
    df["company_profile"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("")
)

texts = df["text"]
labels = df["fraudulent"]

#text processing using NLTK
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

#apply cleaning
cleaned_texts = [clean_text(text) for text in texts]

#Feature Extraction using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(cleaned_texts)

#train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42,stratify=labels
)

#Train the Model (Scikit-learn)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#model evaluation
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, zero_division=0))


#Confusion Matrix Visualization

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

#Word Embeddings using spaCy
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")

def spacy_vector(text):
    return nlp(text).vector


#convert text to vectors
X_spacy = np.array([spacy_vector(text) for text in cleaned_texts])


#user input prediction
def predict_text(user_text):
    cleaned = clean_text(user_text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

while True:
    user_input = input("Enter text (or type exit): ")
    if user_input.lower() == "exit":
        break
    print("Prediction:", predict_text(user_input))

