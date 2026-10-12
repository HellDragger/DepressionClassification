import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import re
import nltk

# Download stopwords if not already available
nltk.download('stopwords')

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

def classification():
    # Load the dataset
    data = pd.read_csv('dataset/Depression_train.csv')

    # Display first few rows to understand the structure
    print(data.head())

    # Assuming 'text' and 'label' columns exist in the dataset
    X = data['text'].apply(preprocess_text)  # Apply preprocessing to the text data
    y = data['label']

    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorizing the text data
    tfidf = TfidfVectorizer(max_features=5000)

    # Transform the text data
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Initialize and train the classifier
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Predicting the test data
    y_pred = model.predict(X_test_tfidf)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    classification()
