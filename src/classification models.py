device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class NN_Deep(nn.Module):
    def __init__(self, drop_out):
        super(NN_Deep, self).__init__()
        self.input_layer = nn.Linear(1000, 500)

        self.h1 = nn.Linear(500, 250)
        self.batch1 = nn.BatchNorm1d(250)

        self.h2 = nn.Linear(250, 125)
        self.batch2 = nn.BatchNorm1d(125)

        self.h3 = nn.Linear(125, 70)
        self.batch3 = nn.BatchNorm1d(70)

        self.h4 = nn.Linear(70, 25)
        self.batch4 = nn.BatchNorm1d(25)

        self.h5 = nn.Linear(25, 5)

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

        return data

def deep_training(model, train_data, train_target, val_data, val_target, learning_rate, num_epochs, batch_size, device=device):
    #trainloader = redefine(train_data, train_target, batch_size)
    #valoader = redefine(val_data, val_target, batch_size)
    model = model.to(device)

    train_size = train_data.shape[0]
    val_size = val_data.shape[0]

    train_batch = train_size // 5
    val_batch = val_size // 5

    classes = np.unique(train_target.values)
    class_weights = compute_class_weight(classes=classes, class_weight='balanced', y=train_target.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for batch_start in range(0, train_size, train_batch):
            batch_end = min(batch_start + train_batch, train_size)
            trainloader = redefine(train_data[batch_start:batch_end], train_target[batch_start:batch_end], batch_size)

            train_loop = tqdm.tqdm(trainloader, desc=f"Train Epoch {epoch+1}", leave=False)
            for data, labels in train_loop:
                optimizer.zero_grad()
                data, labels = data.to(device), labels.to(device)
                logits = model(data)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)

                del data, labels
                torch.cuda.empty_cache()
                gc.collect()

            print(f"Training loss for batch {(batch_start+train_batch) // train_batch}, for epoch {epoch+1} is:\t{running_loss/len(trainloader.dataset)}\n")
            train_losses.append(running_loss/len(trainloader.dataset))

        model.eval()
        with torch.no_grad():

            for batch_start in range(0, val_size, val_batch):
                batch_end = min(batch_start + val_batch, val_size)
                valoader = redefine(val_data[batch_start:batch_end], val_target[batch_start:batch_end], batch_size)

                val_loop = tqdm.tqdm(valoader, desc=f"Validate Epoch {epoch+1}", leave=False)
                val_loss = 0.0

                for data, labels in val_loop:
                    data, labels = data.to(device), labels.to(device)
                    logits = model(data)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * data.size(0)

                    del data, labels
                    gc.collect()

            print(f"Validating loss for batch {(batch_start + val_batch) // val_batch}, for epoch {epoch+1} is:\t{val_loss/len(valoader.dataset)}\n")
            val_losses.append(val_loss/len(valoader.dataset))

    return model, train_losses, val_losses


def deep_predict(model, data, target, batch_size, device = device):
    """Predict with monitoring for memory usage and progress."""
    model.eval()
    model.to(device)
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
                data_slice = np.array(data.iloc[batch_start:batch_end] if hasattr(data, "iloc") else data[batch_start:batch_end])

            # Slice labels
            if hasattr(target, "iloc"):
                target_slice = target.iloc[batch_start:batch_end].copy()
            else:
                target_slice = target[batch_start:batch_end]

            dataloader = redefine(data_slice, target_slice, batch_size)
            data_loop = tqdm.tqdm(dataloader, leave=False, desc=f"Batch {batch_start//data_batch + 1}")

            batch_predictions = []
            for batch_data, labels in data_loop:
                batch_data = batch_data.to(device)
                logits = model(batch_data)
                _, predicted = torch.max(logits, 1)
                if predicted.is_cuda:
                    predicted = predicted.cpu()
                batch_predictions.append(predicted.numpy())

            predictions.extend(np.concatenate(batch_predictions))

            # Memory cleanup
            del data_slice, target_slice, dataloader, batch_predictions
            torch.cuda.empty_cache()
            gc.collect()
            print_memory(f"After batch {batch_start//data_batch + 1}")

    print("✅ Prediction complete.")
    print_memory("Final")
    return np.array(predictions)

