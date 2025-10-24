import pandas as pd
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api

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


def word2vec(texts, embedding_matrix=None, w2v_trained_model=[], word_to_index={}, emb_dim = 200, max_len=None):
    tokenized_texts = [text.lower().split() for text in texts]

    if w2v_trained_model is None or len(w2v_trained_model) == 0:
        print("🔍 Loading pretrained embeddings: glove-wiki-gigaword-200 ...")
        w2v_trained_model = api.load("glove-wiki-gigaword-200")

    if word_to_index is None or len(word_to_index) == 0:
        unique_words = set(word for text in tokenized_texts for word in text)
        vocab_size = len(unique_words)

        word_to_index = {word: i + 2 for i, word in enumerate(w2v_trained_model.key_to_index) if word in unique_words}
        word_to_index["<pad>"] = 0
        word_to_index["<unk>"] = 1
        index_to_word = {i: w for w, i in word_to_index.items()}

    else:
        vocab_size = len(word_to_index)

    if embedding_matrix is None or len(embedding_matrix) == 0:
        embedding_matrix = np.zeros((vocab_size, emb_dim))
        for word, i in word_to_index.items():
            if word in w2v_trained_model:
                embedding_matrix[i] = w2v_trained_model[word]
            else:
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

    encoded_texts = [encode_text(tokens, word_to_index) for tokens in tokenized_texts]

    if max_len is None:
        max_len = max(len(seq) for seq in encoded_texts)  # for training
    else:
        # for validation → cap or pad sequences to training max_len
        encoded_texts = [seq[:max_len] if len(seq) > max_len else seq for seq in encoded_texts]

    padded_texts = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in encoded_texts],
        batch_first=True,
        padding_value=word_to_index["<pad>"]
    )
    np.savetxt('embedding_matrix.csv', embedding_matrix)
    if padded_texts.shape[1] < max_len:
        padded_texts = torch.nn.functional.pad(padded_texts, (0, max_len - padded_texts.shape[1]), value=word_to_index["<pad>"])

    return padded_texts, embedding_matrix, w2v_trained_model, word_to_index, max_len
