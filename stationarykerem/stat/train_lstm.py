# train_lstm.py

import pickle
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x.shape = (batch_size, seq_len, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # shape=(batch_size, hidden_size)
        out = self.fc(last_hidden)  # shape=(batch_size,1)
        return out

def main():
    # 1) Train verisini yükle
    with open("train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    seqs_train = train_data["seqs"]   # list of arrays (farklı uzunlukta)
    y_train_seq = train_data["y_seq"] # shape=(N,)
    
    # 2) Model
    model = LSTMClassifier(input_size=1, hidden_size=16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3) Basit "epoch" loop (batch_size=1 yaklaşımlı, her seriyi tek tek)
    epochs = 5  # Basit örnek olsun
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, seq_array in enumerate(seqs_train):
            # seq_array.shape = (seq_len,)
            # label
            label = y_train_seq[i]
            
            # Tensöre dönüştür: (1, seq_len, 1)
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            lbl_tensor = torch.tensor([label], dtype=torch.float32).unsqueeze(1)  # shape=(1,1)
            
            optimizer.zero_grad()
            out = model(seq_tensor)  # shape=(1,1)
            loss = criterion(out, lbl_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss:.4f}")
    
    # 4) Modeli kaydet
    torch.save(model.state_dict(), "lstm_model.pt")
    print("LSTM modeli eğitildi ve 'lstm_model.pt' olarak kaydedildi.")

if __name__ == "__main__":
    main()
