from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer
from datasets import load_dataset
import pandas as pd
import torch

def get_emotion_dataset():
    return load_dataset('dair-ai/emotion')

def naive_bayes_preprocessing(remove_stopwords=False, use_bigrams=False):
    
    dataset = load_dataset('dair-ai/emotion')
    
    vectorizer = CountVectorizer(
        stop_words='english' if remove_stopwords else None, # rmv stopwords
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

def bert_preprocessing():
    
    dataset = load_dataset('dair-ai/emotion')

    tokenizer = BertTokenizer.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')

    processed_data = {}
    for split, data in dataset.items():
        df = pd.DataFrame(data)

        # Note: max length of data in all splits is 66 words
        tokenized_inputs = tokenizer(df['text'].tolist(), padding='max_length', truncation=True, max_length=70, return_tensors='pt')
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        labels = torch.tensor(df['label'].values)

        processed_data[split] = (input_ids, attention_mask, labels)

    return processed_data, tokenizer