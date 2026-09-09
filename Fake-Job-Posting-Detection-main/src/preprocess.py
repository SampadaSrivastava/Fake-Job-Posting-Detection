import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Text Cleaning Function

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Join words
    return " ".join(words)