from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

# Download the necessary NLTK data for lemmatization
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define the text
text = "Tokenization is the process of breaking down text into individual words or phrases. Lemmatization is similar to stemming but it brings context to the words."

# Tokenization and stop words removal using Gensim
filtered_tokens = [token for token in simple_preprocess(text) if token not in STOPWORDS]
print("Tokens after stop words removal:", filtered_tokens)

# Stemming using NLTK's PorterStemmer (since Gensim does not provide stemming directly)
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)

# Lemmatization using NLTK's WordNetLemmatizer (since Gensim does not provide lemmatization directly)
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)
