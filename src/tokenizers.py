import numpy as np
import torch
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
import gensim.downloader as api
from google.colab import drive
import os
drive.mount('/content/drive')

def TF_IDF(train_data, val_data, train_labels):
    train_data = train_data.astype(str)
    val_data = val_data.astype(str)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, min_df=15)
    X_train = vectorizer.fit_transform(train_data.astype(str))
    X_val = vectorizer.transform(val_data.astype(str))

    selector = SelectKBest(chi2, k=5000)
    X_train = selector.fit_transform(X_train, train_labels)
    X_val   = selector.transform(X_val)

    return X_train, X_val
def encode_text(tokens, word_to_index):
    return [word_to_index.get(word, word_to_index["<unk>"]) for word in tokens]


def word2vec(texts, embedding_matrix=None, w2v_trained_model=None, word_to_index=None, emb_dim = 200, max_len=None):
    texts = texts.astype(str)
    tokenized_texts = [text.lower().split() for text in texts]

    if w2v_trained_model is None:
        print("🔍 Loading pretrained embeddings: glove-wiki-gigaword-200 ...")
        if os.path.exists("drive/MyDrive/glove_200.word2vec"):
            print("Loading model from Drive...")
            w2v_trained_model = KeyedVectors.load_word2vec_format("drive/MyDrive/glove_200.word2vec", binary=True)
        else:
            print("Downloading model...")
            w2v_trained_model = api.load("glove-wiki-gigaword-200")
            w2v_trained_model.save_word2vec_format("drive/MyDrive/glove_200.word2vec", binary=True)

    if word_to_index is None:
        unique_words = set(word for text in tokenized_texts for word in text)
        vocab_size = len(unique_words)

        word_to_index = {word: i + 2 for i, word in enumerate(w2v_trained_model.key_to_index) if word in unique_words}
        word_to_index["<pad>"] = 0
        word_to_index["<unk>"] = 1

    else:
        vocab_size = len(word_to_index)

    if embedding_matrix is None:
        embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
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
        [torch.tensor(seq, dtype=torch.int) for seq in encoded_texts],
        batch_first=True,
        padding_value=word_to_index["<pad>"]
    )
    if padded_texts.shape[1] < max_len:
        padded_texts = torch.nn.functional.pad(padded_texts, (0, max_len - padded_texts.shape[1]), value=word_to_index["<pad>"])

    return padded_texts, embedding_matrix, w2v_trained_model, word_to_index, max_len

