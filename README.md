# text_classification_model
Text Classification Model – Job Fraud Detection

This project implements a text classification model to automatically identify fraudulent job postings using Natural Language Processing (NLP) and Machine Learning techniques.
The model analyzes job-related text such as job title, description, company profile, and requirements to classify each posting as real or fraudulent.

🔧 Technologies Used

Python

NLTK – text preprocessing (tokenization, stopword removal)

Scikit-learn – TF-IDF feature extraction, model training, and evaluation

spaCy – word embeddings (optional)

Matplotlib – confusion matrix visualization

⚙️ Model Workflow

Load job posting data from a CSV file

Clean and preprocess text using NLTK

Convert text into numerical features using TF-IDF

Train a Logistic Regression classifier

Evaluate performance using Precision, Recall, F1-Score, and Confusion Matrix

🎯 Outcome

The model successfully classifies job postings as fraudulent or genuine, demonstrating how NLP and machine learning can be applied to solve real-world text classification problems.
