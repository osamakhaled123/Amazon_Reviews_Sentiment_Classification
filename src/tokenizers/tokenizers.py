import pandas as pd
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sklearn.feature_extraction.text



def TF_IDF(training_data, validating_data, max_features=10000, min_df=15, k=5000):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, min_df=min_df)
    X_train = vectorizer.fit_transform(training_data['cleaned'].astype(str))
    X_val = vectorizer.transform(validating_data['cleaned'].astype(str))

    selector = SelectKBest(chi2, k=k)
    X_train = selector.fit_transform(X_train, training_data['score'])
    X_val = selector.transform(X_val)

    unique, counts = np.unique(training_data['score'].values, return_counts=True)
    class_weights = 1 / counts

    return X_train, X_val, class_weights

def encode_text(tokens, word_to_index):
    return [word_to_index.get(word, word_to_index["<unk>"]) for word in tokens]


def word2vec(texts, labels, w2v_trained_model=[], word_to_index={}, emb_dim=300):
    tokenized_texts = [text.lower().split() for text in texts]
    if not w2v_trained_model:
        w2v_trained_model = KeyedVectors.load_word2vec_format('/path/to/GoogleNews-vectors-negative300.bin',
                                                              binary=True)

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

    return padded_texts


def print_memory(prefix=""):
    """Utility to print memory stats (works in Kaggle + Colab)."""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e9  # GB
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        print(f"{prefix} | 🧠 CPU: {cpu_mem:.2f} GB | ⚡ GPU: {gpu_mem:.2f} GB")
    else:
        print(f"{prefix} | 🧠 CPU: {cpu_mem:.2f} GB")


def redefine(data, labels, batch_size):
    """Prepare DataLoader from various data types."""
    if isinstance(data, scipy.sparse.csr_matrix):
        data_ = torch.tensor(data.toarray(), dtype=torch.float32)
    elif torch.is_tensor(data):
        data_ = data.clone().detach().float()
    else:
        data_ = torch.tensor(np.array(data), dtype=torch.float32)

    if hasattr(labels, "values"):
        labels_ = torch.tensor(labels.values, dtype=torch.long)
    else:
        labels_ = torch.tensor(np.array(labels), dtype=torch.long)

    assert data_.shape[0] == len(labels_), f"Data/Label length mismatch: {data_.shape[0]} vs {len(labels_)}"

    dataset = torch.utils.data.TensorDataset(data_, labels_)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

