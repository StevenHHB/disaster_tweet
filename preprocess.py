import re
from sklearn.feature_extraction.text import TfidfVectorizer

# preprocess data
# remove missing values, handle extreme values, turn to lowercase, remove special characters, stemming
# tokenization, vectorization, remove stop words


class TextPreprocess:
    def __init__(self, stop_words='english', max_features=None):
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words, max_features=max_features)

    def clean_text(self, texts):
        text = texts.lower()
        text = re.sub(r'[^a-z]', ' ', text)  # remove special characters
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove urls
        return text.strip()

    # apply to multiple texts
    def preprocess_text(self, texts):
        return [self.clean_text(text) for text in texts]

    # fit into vectorizer
    def fit_transform(self, texts):
        cleaned_texts = self.preprocess_text(texts)
        return self.vectorizer.fit_transform(cleaned_texts)

    # vectorize new text
    def transform(self, texts):
        cleaned_texts = self.preprocess_text(texts)
        return self.vectorizer.transform(cleaned_texts)
