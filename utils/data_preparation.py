from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def get_emotion_dataset():
    return load_dataset('dair-ai/emotion')

def naive_bayes_preprocessing(remove_stopwords=False, use_bigrams=False):
    
    dataset = load_dataset('dair-ai/emotion')
    
    vectorizer = CountVectorizer(
        stop_words='english' if remove_stopwords else None, # remove stopwords
        ngram_range=(1, 2) if use_bigrams else (1, 1)       # use bigrams
    )

    processed_data = {}
    for split, data in dataset.items():
        df = pd.DataFrame(data)
            
        # X is a very sparse matrix
        X = vectorizer.fit_transform(df['text']) if split == 'train' else vectorizer.transform(df['text'])
        y = df['label'].values
        
        processed_data[split] = (X, y)
    
    return processed_data, vectorizer