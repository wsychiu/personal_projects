#import re
import os
import pandas as pd
import datetime

import spacy
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing for ML model
def clean_text(text):
    # Cleaning data to remove tags (@, #) and websites
    re_pattern = r'@\w+|#\w+'
    re_pattern += r'|https?://\S+|www\.\S+'
    normalized_text = text.str.replace(
        re_pattern, '', regex=True
    )
    
    # Removing extra whitespace
    normalized_text = normalized_text.str.replace(
        '\s+', ' ', regex=True
    )

    return normalized_text

def apply_lemmatization(df, batch_size):
    spacy_model = spacy.load('en_core_web_sm')

    # Removing punctuation and stop words, lemmatizing
    # Disabling certain processing - parses and Named Entity Recog. not required
    disabled = ['parser', 'ner']
    lemmatized_title = spacy_model.pipe(
        df['normalized_title'],
        disable=disabled,
        batch_size=batch_size
    )

    df['normalized_title'] = [
        ' '.join(
            token.lemma_.lower()
            for token in text
            if not token.is_stop and not token.is_punct
        )
        for text in lemmatized_title
    ]

    lemmatized_text = spacy_model.pipe(
        df['normalized_text'],
        disable=disabled,
        batch_size=batch_size
    )

    df['normalized_text'] = [
        ' '.join(
            token.lemma_.lower()
            for token in text
            if not token.is_stop and not token.is_punct
        )
        for text in lemmatized_text
    ]

    return df

def feature_extraction(df):
    # Adding counts of @/# tags and URLs
    re_pattern = r'@\w+|#\w+'
    re_pattern += r'|https?://\S+|www\.\S+'
    df['tag_url_count'] = df['text'].str.count(re_pattern)

    return df

# Dropping all nulls and duplicate rows and duplicated articles
def basic_data_clean(df):
    # Removing records where there is no text in article
    empty_text = df['text'].str.strip().str.len()==0
    return (
        df.drop_duplicates(subset=['title', 'text'], keep='first')
        .drop(df[empty_text].index)
        .dropna(how='all')
        .drop_duplicates()
        .copy()
    )


def data_preprocessing(df, lemma, batch_size):
    df_cleaned = basic_data_clean(df)
    
    # Removing subject to prevent target leakage
    df_cleaned = df_cleaned.drop(columns='subject')    
        
    # Enforcing data types
    df_cleaned = df_cleaned.astype({
        'title' : 'string',
        'text' : 'string',
        'date' : 'datetime64[ns]',
    })

    # Feature Extractions
    df_cleaned = feature_extraction(df_cleaned)
    
    # Text transformations
    df_cleaned['normalized_title'] = clean_text(df_cleaned['title'])
    df_cleaned['normalized_text'] = clean_text(df_cleaned['text'])

    # Lemmatization (if required)
    if lemma: 
        print("Lemmatizing...")    
        df_cleaned = apply_lemmatization(df_cleaned, batch_size)
        print("Lemma complete")
    
    return df_cleaned

class TextProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, lemma=False, batch_size=500):
        self.lemma = lemma
        self.batch_size = batch_size
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return data_preprocessing(
            X.copy(),
            lemma=self.lemma,
            batch_size=self.batch_size,
        )

def get_timestamp(format='%Y%m%d_%H%M%S'):
    return str(datetime.datetime.now().strftime(format))

def load_data(fpath_data):
    data_ext = os.path.splitext(fpath_data)[1].lower()

    if data_ext == '.csv':
        return pd.read_csv(fpath_data)

    raise ValueError(f'Unsupported file type: {data_ext}')