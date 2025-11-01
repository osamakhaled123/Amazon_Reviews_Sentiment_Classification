import numpy as np
import torch
from torch import nn
from transformers import get_scheduler
from torch.optim import AdamW
import scipy.sparse
import psutil
import os
from sklearn.utils.class_weight import compute_class_weight
import tqdm
import gc
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_memory(prefix=""):
    """Utility to print memory stats (works in Kaggle + Colab)."""
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e9
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
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class NN_Deep(nn.Module):
    def __init__(self, drop_out):
        super(NN_Deep, self).__init__()
        self.input_layer = nn.Linear(1753, 1000)

        self.h1 = nn.Linear(1000, 500)
        self.batch1 = nn.BatchNorm1d(500)

        self.h2 = nn.Linear(500, 250)
        self.batch2 = nn.BatchNorm1d(250)

        self.h3 = nn.Linear(250, 125)
        self.batch3 = nn.BatchNorm1d(125)

        self.h4 = nn.Linear(125, 70)
        self.batch4 = nn.BatchNorm1d(70)

        self.h5 = nn.Linear(70, 25)
        self.batch5 = nn.BatchNorm1d(25)

        self.h6 = nn.Linear(25, 5)

        self.drop_out = nn.Dropout(p=drop_out)
        self.activation = nn.ReLU()

    def forward(self, data):
        data = self.input_layer(data)

        data = self.h1(data)
        data = self.batch1(data)
        data = self.activation(data)
        data = self.drop_out(data)

        data = self.h2(data)
        data = self.batch2(data)
        data = self.activation(data)
        data = self.drop_out(data)

        data = self.h3(data)
        data = self.batch3(data)
        data = self.activation(data)
        data = self.drop_out(data)

        data = self.h4(data)
        data = self.batch4(data)
        data = self.activation(data)
        data = self.drop_out(data)

        data = self.h5(data)
        data = self.batch5(data)
        data = self.activation(data)
        data = self.drop_out(data)

        data = self.h6(data)

        return data


def deep_training(model, train_data, train_target, val_data, val_target, learning_rate, num_epochs, batch_size, device):
    model = model.to(device)

    train_size = train_data.shape[0]
    val_size = val_data.shape[0]

    train_batch = train_size // 5
    val_batch = val_size // 5

    classes = np.unique(train_target.values)
    class_weights = compute_class_weight(classes=classes, class_weight='balanced', y=train_target.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_start in range(0, train_size, train_batch):
            model.train()
            batch_end = min(batch_start + train_batch, train_size)
            trainloader = redefine(train_data[batch_start:batch_end], train_target[batch_start:batch_end], batch_size)

            train_loop = tqdm.tqdm(trainloader, desc=f"Train Epoch {epoch + 1}", leave=False)
            for data, labels in train_loop:
                optimizer.zero_grad()
                data, labels = data.to(device), labels.to(device)
                logits = model(data)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)

            print(
                f"Training loss for batch {(batch_start + train_batch) // train_batch}, for epoch {epoch + 1} is:\t{running_loss / len(trainloader.dataset)}\n")
            train_losses.append(running_loss / len(trainloader.dataset))

            del train_loop, trainloader
            torch.cuda.empty_cache()
            gc.collect()

        model.eval()
        with torch.no_grad():

            for batch_start in range(0, val_size, val_batch):
                batch_end = min(batch_start + val_batch, val_size)
                valoader = redefine(val_data[batch_start:batch_end], val_target[batch_start:batch_end], batch_size)

                val_loop = tqdm.tqdm(valoader, desc=f"Validate Epoch {epoch + 1}", leave=False)
                val_loss = 0.0

                for data, labels in val_loop:
                    data, labels = data.to(device), labels.to(device)
                    logits = model(data)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * data.size(0)

                print(
                    f"Validating loss for batch {(batch_start + val_batch) // val_batch}, for epoch {epoch + 1} is:\t{val_loss / len(valoader.dataset)}\n")
                val_losses.append(val_loss / len(valoader.dataset))

                del val_loop, valoader
                torch.cuda.empty_cache()
                gc.collect()

    return model, train_losses, val_losses


def deep_predict(model, data, target, batch_size):
    """Predict with monitoring for memory usage and progress."""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    predictions = []

    data_size = data.shape[0]
    data_batch = data_size // 5 or batch_size  # fallback for small data

    print(f"🚀 Starting prediction on {data_size} samples.")
    print_memory("Initial")

    with torch.no_grad():
        for batch_start in range(0, data_size, data_batch):
            batch_end = min(batch_start + data_batch, data_size)

            # Slice data safely
            if isinstance(data, scipy.sparse.csr_matrix):
                data_slice = data[batch_start:batch_end].copy()
            elif torch.is_tensor(data):
                data_slice = data[batch_start:batch_end].clone().detach()
            else:
                data_slice = np.array(
                    data.iloc[batch_start:batch_end] if hasattr(data, "iloc") else data[batch_start:batch_end])

            # Slice labels
            if hasattr(target, "iloc"):
                target_slice = target.iloc[batch_start:batch_end].copy()
            else:
                target_slice = target[batch_start:batch_end]

            dataloader = redefine(data_slice, target_slice, batch_size)
            data_loop = tqdm.tqdm(dataloader, leave=False, desc=f"Batch {batch_start // data_batch + 1}")

            batch_predictions = []
            for batch_data, labels in data_loop:
                batch_data = batch_data.to(device)
                labels = labels.to(device)
                logits = model(batch_data)
                _, predicted = torch.max(logits, 1)
                if predicted.is_cuda:
                    predicted = predicted.cpu()
                batch_predictions.append(predicted.numpy())

            predictions.extend(np.concatenate(batch_predictions))

            # Memory cleanup
            print_memory(f"After batch {batch_start // data_batch + 1}")
            del data_slice, target_slice, dataloader, data_loop
            torch.cuda.empty_cache()
            gc.collect()

    print("✅ Prediction complete.")
    print_memory("Final")
    return np.array(predictions)


class GRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False
        )
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.int()
        x = self.embedding(x)
        _, h = self.gru(x)
        h = self.dropout(h[-1])
        out = self.fc(h)
        return out


def GRU_train(model, train_data, train_target, val_data, val_target, learning_rate, num_epochs, batch_size, device):
    model = model.to(device)

    train_size = train_data.shape[0]
    val_size = val_data.shape[0]

    train_batch = train_size // 10
    val_batch = val_size // 10

    classes = np.unique(train_target.values)
    class_weights = compute_class_weight(classes=classes, class_weight='balanced', y=train_target.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()

        for batch_start in range(0, train_size, train_batch):
            batch_end = min(batch_start + train_batch, train_size)
            trainloader = redefine(train_data[batch_start:batch_end], train_target[batch_start:batch_end], batch_size)

            train_loop = tqdm.tqdm(trainloader, desc=f"Train Epoch {epoch + 1}", leave=False)
            total_loss = 0.0

            for X_batch, y_batch in train_loop:
                X_batch, y_batch = X_batch.to(device), y_batch.int().to(device)
                optimizer.zero_grad()

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.long().to(device))
                loss.backward()

                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)

            print(
                f"Training loss for batch {(batch_start + train_batch) // train_batch}, for epoch {epoch + 1} is:\t{total_loss / len(trainloader.dataset):.4f}\n")
            train_losses.append(total_loss / len(trainloader.dataset))

            del train_loop, trainloader
            torch.cuda.empty_cache()
            gc.collect()

        model.eval()
        with torch.no_grad():
            for batch_start in range(0, val_size, val_batch):
                batch_end = min(batch_start + val_batch, val_size)
                valoader = redefine(val_data[batch_start:batch_end], val_target[batch_start:batch_end], batch_size)

                val_loop = tqdm.tqdm(valoader, desc=f"Validate Epoch {epoch + 1}", leave=False)
                val_loss = 0.0

                for X_batch, y_batch in val_loop:
                    X_batch, y_batch = X_batch.to(device), y_batch.int().to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.long().to(device))
                    val_loss += loss.item() * X_batch.size(0)

                print(
                    f"Validating loss for batch {(batch_start + val_batch) // val_batch}, for epoch {epoch + 1} is:\t{val_loss / len(valoader.dataset):.4f}\n")
                val_losses.append(val_loss / len(valoader.dataset))

                del val_loop, valoader
                torch.cuda.empty_cache()
                gc.collect()

    return model, train_losses, val_losses


def GRU_predict(model, X_batch, y_batch, batch_size):
    model.eval()
    model.to(device)
    predictions = []

    val_size = len(X_batch)
    val_batch = max(1, val_size // 5)

    print(f"🚀 Predicting {val_size} samples (batch size {batch_size})")

    with torch.no_grad():
        for batch_start in range(0, val_size, val_batch):
            batch_end = min(batch_start + val_batch, val_size)
            X_slice = X_batch[batch_start:batch_end]

            # handle labels
            if hasattr(y_batch, "iloc"):
                y_slice = y_batch.iloc[batch_start:batch_end]
            else:
                y_slice = y_batch[batch_start:batch_end]

            if len(X_slice) != len(y_slice):
                print(f"⚠️ Skipping mismatch batch ({len(X_slice)} vs {len(y_slice)})")
                continue

            valoader = redefine(X_slice, y_slice, batch_size)
            val_loop = tqdm.tqdm(valoader, desc=f"Validate batch {(batch_start + val_batch) / val_batch}", leave=False)
            for Xb, _ in val_loop:
                Xb = Xb.to(device)
                logits = model(Xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)

            torch.cuda.empty_cache()
            gc.collect()

    return np.array(predictions)


def DistilBert_train(model, train_dataloader, val_dataloader, device, num_epochs=4, learning_rate=5e-5):
    train_losses, val_losses = [], []

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()

            running_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

        del batch
        gc.collect()

        print(f"Epoch {epoch + 1} | Loss: {running_loss / len(train_dataloader.dataset):.4f}")
        train_losses.append(running_loss / len(train_dataloader.dataset))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                loss = output.loss
                val_loss += loss.item()

            del batch
            gc.collect()

            print(f"Epoch {epoch + 1} | Loss: {val_loss / len(val_dataloader.dataset):.4f}")
            val_losses.append(val_loss / len(val_dataloader.dataset))

    return model, train_losses, val_losses


def DistilBert_predict(model, val_dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["score"].cpu().numpy())

        del batch
        gc.collect()
    return predictions, true_labels


def plot_losses(train_losses, val_losses):
    #epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")

    # Highlight min val loss
    min_val_idx = val_losses.index(min(val_losses))
    plt.scatter(min_val_idx + 1, val_losses[min_val_idx], color='green', s=100, label='Best Validation Loss')

    plt.title('Training vs Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.xticks(range(1, len(train_losses)+1))
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def smooth_curve(points, factor=0.9):
    smoothed = []
    for p in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed