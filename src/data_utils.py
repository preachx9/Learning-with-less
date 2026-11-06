"""First experiment: audit Yorùbá data and evaluate two fixed baselines on dev.

Run: python start_project.py
This script deliberately does not download or evaluate the final test split.
"""
from pathlib import Path
import hashlib
import json
import platform
import time
import unicodedata
import urllib.request
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
REVISION = 'fa3b5fff8a91d187bf0c5900a39c4271d08cf7fe'
BASE = f'https://huggingface.co/datasets/masakhane/masakhanews/resolve/{REVISION}/data/yor/'

def normalise(text):
    return ' '.join(unicodedata.normalize('NFC', str(text)).lower().split())

def word_tokens(text):
    # Preserve Yorùbá combining tone marks within words. Python's default
    # word regex can split at combining marks, even after NFC normalisation.
    words, current = [], []
    for c in text:
        if unicodedata.category(c)[0] in 'LMN':
            current.append(c)
        elif current:
            words.append(''.join(current))
            current = []
    if current:
        words.append(''.join(current))
    return words

def load_split(name):
    path = ROOT / 'data' / f'{name}.tsv'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(BASE + path.name, timeout=90) as response:
            path.write_bytes(response.read())
    df = pd.read_csv(path, sep='\t').fillna('')
    required = {'text', 'headline', 'category', 'url'}
    assert required.issubset(df.columns), df.columns
    df['input'] = df['text'].map(normalise)
    return df, hashlib.sha256(path.read_bytes()).hexdigest()
