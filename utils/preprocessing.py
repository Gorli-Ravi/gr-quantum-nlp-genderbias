import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    df = pd.read_csv(path)
    return df['sentence'], df['label']

def vectorize_text(sentences, max_features=8):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(sentences).toarray()
    return X

