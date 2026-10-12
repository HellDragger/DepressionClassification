import nltk
import pandas as pd
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset
dataset = "dataset/Depression_train.csv"
data = pd.read_csv(dataset)
print("Top five sentences:-")
print(data.head())

# Check for missing values
print("Missing values in the dataset:")
print(data.isna().sum())

# Load stopwords once; looking them up per token makes preprocessing take minutes
stop_words = set(stopwords.words('english'))

# Data Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    
    # Join the cleaned tokens back into a string
    return ' '.join(tokens)

# Apply preprocessing to the 'text' column
if 'text' in data.columns:  # Ensure there's a 'text' column to process
    data['processed_text'] = data['text'].apply(preprocess_text)

# Print the first 5 preprocessed sentences
print("\nPreprocessed sentences:")
print(data[['text', 'processed_text']].head())
