import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Parametreler
SERIES_FOLDER = "data/series"
MAX_LEN = 150
BATCH_SIZE = 64
EPOCHS = 30

# Veriyi oku
metadata = pd.read_csv("data/series_metadata.csv")
label_encoder = LabelEncoder()
metadata["label_encoded"] = label_encoder.fit_transform(metadata["label"])

train_meta = metadata[metadata["split"] == "train"]
test_meta = metadata[metadata["split"] == "test"]

def load_and_pad(filename):
    series = pd.read_csv(os.path.join(SERIES_FOLDER, filename), header=None).squeeze("columns").values
    padded = np.pad(series, (0, MAX_LEN - len(series)), 'constant')
    return padded

class TimeSeriesDataset(Dataset):
    def __init__(self, metadata):
        self.X = [load_and_pad(fname) for fname in metadata["filename"]]
        self.y = metadata["label_encoded"].tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1), torch.tensor(self.y[idx], dtype=torch.long)

# Dataset ve Dataloader
train_dataset = TimeSeriesDataset(train_meta)
test_dataset = TimeSeriesDataset(test_meta)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model tanımı
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

# Test
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        preds = torch.argmax(output, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
