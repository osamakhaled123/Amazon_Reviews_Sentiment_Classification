import pandas as pd
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

def TF_IDF(training_data, validating_data, max_features=10000, min_df=15, k=5000):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, min_df=min_df)
    X_train = vectorizer.fit_transform(training_data['cleaned'].astype(str))
    X_val = vectorizer.transform(validating_data['cleaned'].astype(str))

    selector = SelectKBest(chi2, k=k)
    X_train = selector.fit_transform(X_train, training_data['score'])
    X_val = selector.transform(X_val)

    return X_train, X_val
def encode_text(tokens, word_to_index):
    return [word_to_index.get(word, word_to_index["<unk>"]) for word in tokens]


def word2vec(texts, w2v_trained_model=[], word_to_index={}, emb_dim=300):
    tokenized_texts = [text.lower().split() for text in texts]
    if not w2v_trained_model:
        w2v_trained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    if not word_to_index:
        unique_words = set()
        for text in tokenized_texts:
            unique_words.update(text)

        vocab_size = len(unique_words)

        word_to_index = {word: i + 2 for i, word in enumerate(w2v_trained_model.wv.index_to_key) if
                         word in unique_words}
        word_to_index["<pad>"] = 0
        word_to_index["<unk>"] = 1
        index_to_word = {i: w for w, i in word_to_index.items()}

    embedding_matrix = np.zeros((vocab_size, emb_dim))

    for word, i in word_to_index.items():
        if word in w2v_trained_model.wv:
            embedding_matrix[i] = w2v_trained_model.wv[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

    encoded_texts = [encode_text(tokens, word_to_index) for tokens in tokenized_texts]
    padded_texts = pad_sequence([torch.tensor(seq) for seq in encoded_texts],
                                batch_first=True,
                                padding_value=word_to_index["<pad>"])

    return padded_texts, embedding_matrix, w2v_trained_model, word_to_index


