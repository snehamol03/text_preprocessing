import spacy
import nltk

# Download the necessary NLTK data for stemming (only needed if you use stemming)
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the text
text = "Tokenization is the process of breaking down text into individual words or phrases. Lemmatization is similar to stemming but it brings context to the words."

# Tokenization, stop words removal, and lemmatization using spaCy
doc = nlp(text)
tokens = [token.text for token in doc]
filtered_tokens = [token for token in doc if not token.is_stop]
lemmatized_tokens = [token.lemma_ for token in filtered_tokens]

print("Tokens:", tokens)
print("Tokens after stop words removal:", [token.text for token in filtered_tokens])
print("Lemmatized Tokens:", lemmatized_tokens)

# For stemming, we need to use NLTK as spaCy does not provide stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token.text) for token in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)
