import pandas as pd
import re
from transformers import MarianMTModel, MarianTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained model for English to French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function for basic text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to string
    return ' '.join(filtered_text)

# Function to translate text using the model and tokenizer
def translate_text(texts, model, tokenizer, batch_size=10, max_length=512):
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        translated = model.generate(**inputs)
        translated_texts.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return translated_texts

# Example: Load a dataset with a text field (CSV with 'text' column)
# You can replace this with your actual dataset
df = pd.read_csv("dataset/Depression_train.csv")  # Replace with your dataset path

# Show the first few rows of the dataset
print("Original Dataset:\n", df.head())

# Apply preprocessing to the 'text' column
df['preprocessed_text'] = df['text'].apply(preprocess_text)

# Translate the preprocessed 'text' column from English to French
df['translated_text'] = translate_text((df['preprocessed_text'].head(1000)).tolist(), model, tokenizer, batch_size=10)

# Show the first few rows of the preprocessed and translated dataset
print("\nPreprocessed and Translated Dataset:\n", df[['text', 'preprocessed_text', 'translated_text']].head())

# Save the dataset with translations to a new CSV file
df.to_csv("translated_dataset.csv", index=False)

print("\nTranslation Completed and Saved!")
