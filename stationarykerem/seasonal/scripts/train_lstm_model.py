# scripts/train_lstm_model.py
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import prepare_lstm_sequences
from utils.visualization import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_precision_recall_curve
)

class LSTMModel(nn.Module):
    """
    PyTorch LSTM modeli.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Son zaman adımının çıktısını al
        out = self.fc(lstm_out[:, -1, :])
        out = self.sigmoid(out)
        return out

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50, patience=10, device=None):
    """
    LSTM modelini eğit.
    
    Args:
        model: PyTorch modeli
        train_loader: Eğitim veri yükleyicisi
        valid_loader: Doğrulama veri yükleyicisi
        criterion: Kayıp fonksiyonu
        optimizer: Optimizasyon algoritması
        num_epochs (int): Eğitim epoch sayısı
        patience (int): Erken durma sabırsızlık sayısı
        device: Cihaz (CPU/GPU)
        
    Returns:
        dict: Eğitim geçmişi
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Eğitim geçmişi
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # En iyi model ağırlıkları
    best_val_loss = float('inf')
    best_model_weights = None
    counter = 0
    
    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Gradyanları sıfırla
            optimizer.zero_grad()
            
            # İleri geçiş
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Geri yayılım
            loss.backward()
            optimizer.step()
            
            # İstatistikleri güncelle
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # İleri geçiş
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # İstatistikleri güncelle
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Geçmişi güncelle
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
        
        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
        
        # Erken durma
        if counter >= patience:
            print(f'Erken durma. Epoch: {epoch+1}')
            break
    
    # En iyi ağırlıkları yükle
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    
    return model, history

def main(args):
    print("LSTM modeli eğitiliyor (PyTorch)...")
    
    # Veri klasörleri
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'lstm_model')
    
    os.makedirs(model_dir, exist_ok=True)
    
    # PyTorch için rastgele tohum ayarla
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Cihazı belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz kullanımı: {device}")
    
    # Zaman serilerini yükle
    print("Veriler yükleniyor...")
    all_series_df = pd.read_csv(os.path.join(raw_dir, 'all_series.csv'))
    
    # Etiketleri yükle
    seasonality_labels = pd.read_csv(os.path.join(processed_dir, 'seasonality_labels.csv'))
    
    # Eğitim/test indekslerini yükle
    train_indices = np.load(os.path.join(processed_dir, 'train_indices.npy'))
    test_indices = np.load(os.path.join(processed_dir, 'test_indices.npy'))
    
    # Tüm serileri bir sözlük olarak grupla
    series_dict = {}
    for series_id, group in all_series_df.groupby('series_id'):
        series_dict[series_id] = pd.Series(group['y'].values, index=pd.to_datetime(group['ds']))
    
    # Eğitim ve test serilerini ayır
    train_series = [series_dict[i] for i in train_indices if i in series_dict]
    test_series = [series_dict[i] for i in test_indices if i in series_dict]
    
    print(f"Eğitim seti boyutu: {len(train_series)}")
    print(f"Test seti boyutu: {len(test_series)}")
    
    # Eğitim ve test etiketleri
    train_seasonality_labels = seasonality_labels[seasonality_labels['series_id'].isin(train_indices)]
    test_seasonality_labels = seasonality_labels[seasonality_labels['series_id'].isin(test_indices)]
    
    # LSTM için sekans verileri hazırla
    print("LSTM sekansları hazırlanıyor...")
    
    seq_length = args.seq_length
    
    # Eğitim verileri
    X_train = []
    y_train = []
    
    for i, ts in enumerate(train_series):
        # ID'yi bul
        series_id = train_indices[i]
        
        # Mevsimsellik etiketi
        is_seasonal = train_seasonality_labels[train_seasonality_labels['series_id'] == series_id]['is_seasonal'].values[0]
        
        # Zaman serisini normalize et
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
        
        # Sekansları oluştur
        for j in range(0, len(normalized_data) - seq_length, args.step):
            X_train.append(normalized_data[j:j+seq_length])
            y_train.append(is_seasonal)
    
    # Test verileri
    X_test = []
    y_test = []
    
    for i, ts in enumerate(test_series):
        # ID'yi bul
        series_id = test_indices[i]
        
        # Mevsimsellik etiketi
        is_seasonal = test_seasonality_labels[test_seasonality_labels['series_id'] == series_id]['is_seasonal'].values[0]
        
        # Zaman serisini normalize et
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
        
        # Sekansları oluştur
        for j in range(0, len(normalized_data) - seq_length, args.step):
            X_test.append(normalized_data[j:j+seq_length])
            y_test.append(is_seasonal)
    
    # PyTorch Tensor'larına dönüştür
    X_train = np.array(X_train).reshape(-1, seq_length, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, seq_length, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    print(f"Eğitim sekansları boyutu: {X_train.shape}")
    print(f"Test sekansları boyutu: {X_test.shape}")
    
    # Eğitim ve doğrulama setlerini ayır
    val_size = int(0.2 * len(X_train_tensor))
    train_size = len(X_train_tensor) - val_size
    
    train_dataset = TensorDataset(X_train_tensor[:train_size], y_train_tensor[:train_size])
    val_dataset = TensorDataset(X_train_tensor[train_size:], y_train_tensor[train_size:])
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # LSTM modelini oluştur
    print("LSTM modeli oluşturuluyor...")
    model = LSTMModel(
        input_size=1,
        hidden_size=args.lstm_units,
        num_layers=2,
        dropout=args.dropout_rate
    )
    
    # Özeti göster
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parametreleri: {num_params}")
    print(model)
    
    # Kayıp fonksiyonu ve optimizasyon
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Modeli eğit
    print("Model eğitiliyor...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=args.epochs,
        patience=10,
        device=device
    )
    
    # Eğitim geçmişini kaydet
    with open(os.path.join(model_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Eğitim geçmişi grafiği
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Plot klasörünü oluştur
    plot_dir = os.path.join(model_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plot_dir, 'training_history.png'))
    plt.close()
    
    # Modeli değerlendir
    print("Model değerlendiriliyor...")
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = outputs.cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(labels.numpy())
    
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred)
    
    print("\nMevsimsellik Tespit Sonuçları:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Sonuçları kaydet
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    with open(os.path.join(model_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Grafikler
    # Karışıklık matrisi
    plot_confusion_matrix(
        y_true, 
        y_pred_binary, 
        classes=['Non-Seasonal', 'Seasonal'],
        title='Seasonality Detection Confusion Matrix (LSTM)',
        save_path=os.path.join(plot_dir, 'confusion_matrix.png')
    )
    
    # ROC eğrisi
    plot_roc_curve(
        y_true,
        y_pred,
        title='ROC Curve for Seasonality Detection (LSTM)',
        save_path=os.path.join(plot_dir, 'roc_curve.png')
    )
    
    # Hassasiyet-Duyarlılık eğrisi
    plot_precision_recall_curve(
        y_true,
        y_pred,
        title='Precision-Recall Curve for Seasonality Detection (LSTM)',
        save_path=os.path.join(plot_dir, 'precision_recall_curve.png')
    )
    
    # Modeli kaydet
    torch.save(model.state_dict(), os.path.join(model_dir, 'seasonality_model.pth'))
    
    # Model mimarisini de kaydet
    model_info = {
        'input_size': 1,
        'hidden_size': args.lstm_units,
        'num_layers': 2,
        'dropout_rate': args.dropout_rate
    }
    
    with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)
    
    print("LSTM modeli eğitimi tamamlandı!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM modelini eğit (PyTorch)')
    parser.add_argument('--seq-length', type=int, default=50, help='Sekans uzunluğu')
    parser.add_argument('--step', type=int, default=10, help='Sekans adım boyutu')
    parser.add_argument('--lstm-units', type=int, default=64, help='LSTM birim sayısı')
    parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout oranı')
    parser.add_argument('--epochs', type=int, default=50, help='Eğitim epoch sayısı')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch boyutu')
    parser.add_argument('--seed', type=int, default=42, help='Rastgele sayı üreteci tohumu')
    
    args = parser.parse_args()
    main(args)