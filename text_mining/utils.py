import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path


resources_to_download = {
    'wordnet': 'corpora/wordnet',
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'omw-1.4': 'corpora/omw-1.4'
}

for resource_name, resource_path in resources_to_download.items():
    try:
        nltk.data.find(resource_path)
        print(f"NLTK resource '{resource_name}' found.")
    except LookupError:
        print(f"NLTK resource '{resource_name}' not found. Downloading...")
        nltk.download(resource_name)
        print(f"NLTK resource '{resource_name}' downloaded.")


DATA_DIR = Path(__file__).resolve().parent / "data"

def load_races():
    try:
        df_races = pd.read_csv(DATA_DIR / "races.csv", sep=';')
        print("Races data loaded successfully.")
        return df_races
    except FileNotFoundError:
        print(f"Error: races.csv not found in {DATA_DIR}")
        return pd.DataFrame()

def load_lotr_texts():
    books = {
        "Fellowship": "01 Fellowship.txt",
        "Two Towers": "02 Two Towers.txt",
        "Return of the King": "03 Return of the King.txt"
    }
    texts = {}
    all_text_corpus = []
    print("Loading LOTR texts...")
    for book_name, filename in books.items():
        try:
            
            with open(DATA_DIR / filename, 'r', encoding='cp1252') as f:
                content = f.read()
                texts[book_name] = content
                all_text_corpus.append(content)
                print(f"Loaded: {filename} ({len(content)} characters)")
        except FileNotFoundError:
            print(f"Error: {filename} not found in {DATA_DIR}")
            texts[book_name] = ""
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            texts[book_name] = ""
    return texts, " ".join(all_text_corpus)

def preprocess_text(text, language='english'):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    tokens = word_tokenize(text)

    try:
        stop_words_set = set(stopwords.words(language))
    except LookupError:
        print(f"Stopwords for '{language}' not found. Please ensure it's downloaded or NLTK has access.")
        stop_words_set = set() 

    filtered_tokens = [word for word in tokens if word not in stop_words_set and len(word) > 2] 

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmatized_tokens

def create_ngrams(tokens, n=2):
    if not tokens or len(tokens) < n:
        return []
    return list(nltk.ngrams(tokens, n))
