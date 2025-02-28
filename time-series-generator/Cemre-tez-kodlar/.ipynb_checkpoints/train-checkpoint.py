import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle 
from models import ConvModel, LSTMModel, ConvLSTMModel
from data_utils import create_loaders
from tqdm import tqdm
from train_utils import calculate_metrics
import os
import wandb
wandb.login()

cfg = {
    "model": "lstm",
    "lr": 0.001,
    "num_epochs": 100,
    "batch_size": 128,
    "hidden_size": 64,
    "num_layers": 3,
    "k_meanshift": 1,
    "k_varshift": 1,
    "k_trendshift": 1,
    "k_anomaly": 1,
    "lr_scheduler_patience": 1,
    "early_stop_patience": 3,
    
}

DATASET_PATH = "data"

#with wandb.init(project="cemretez_run", config=cfg, mode="disabled") as run:
with wandb.init(project="cemretez_run", config=cfg) as run:
    cfg = wandb.config

    BEST_MODEL_PATH = os.path.join(wandb.run.dir, "model.h5")   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    train_paths = [f'{DATASET_PATH}/train_{length}.pkl' for length in ['short', 'medium', 'long']]
    val_paths = [f'{DATASET_PATH}/val_{length}.pkl' for length in ['short', 'medium', 'long']]
    
    train_loaders = create_loaders(train_paths, batch_size=cfg.batch_size)
    val_loaders = create_loaders(val_paths, batch_size=cfg.batch_size * 2, shuffle = False)
        
    criterion = nn.BCEWithLogitsLoss()
    
    if cfg.model == "conv":
        model = ConvModel(input_size=1, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers).to(device)
    elif cfg.model == "lstm":
        model = LSTMModel(1, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, device=device).to(device)
    elif cfg.model == "convlstm":
        model = ConvLSTMModel(1, hidden_size = cfg.hidden_size, num_layers=cfg.num_layers, device=device).to(device)
    else:
        raise NotImplementedError

        
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.lr_scheduler_patience, mode="max", factor=0.95) 
    best_val_f1 = 0
    epochs_without_improvement = 0

    iterations = 0
    for epoch in range(cfg.num_epochs):
        print(f'Epoch {epoch} is started.')
        val_metrics = calculate_metrics(model, val_loaders, device)
        scheduler.step(val_metrics["avg_f1"])
        wandb.log(val_metrics)

        
        #early stopping
        if val_metrics["avg_f1"] > best_val_f1 * (1001/1000):
            best_val_f1 = val_metrics["avg_f1"]
            epochs_without_improvement = 0
            torch.save(model, BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= cfg.early_stop_patience and epoch > 10:
            break
        for train_loader in train_loaders:
            for batch in train_loader:
                data, mean_shift, variance_shift, trend_shift, anomaly = batch
                
                data = data.to(device)
                mean_shift = mean_shift.to(device)
                variance_shift = variance_shift.to(device)
                trend_shift = trend_shift.to(device)
                anomaly = anomaly.to(device)
                
                output = model(data)
            
                mean_shift_pred = output[:, 0, :]
                variance_shift_pred = output[:, 1, :]
                trend_shift_pred = output[:, 2, :]
                anomaly_pred = output[:, 3, :]
            
                mean_shift_loss = criterion(mean_shift_pred, mean_shift) * cfg.k_meanshift
                variance_shift_loss = criterion(variance_shift_pred, variance_shift) *  cfg.k_varshift 
                trend_shift_loss = criterion(trend_shift_pred, trend_shift) * cfg.k_trendshift
                anomaly_loss = criterion(anomaly_pred, anomaly) * cfg.k_anomaly
        
                total_loss = (mean_shift_loss + variance_shift_loss + trend_shift_loss + anomaly_loss)
                total_loss /= cfg.k_meanshift + cfg.k_varshift + cfg.k_trendshift + cfg.k_anomaly
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if (iterations % 10) == 0:
                    wandb.log({
                        "train_loss": total_loss.item(),
                        "mean_shift_loss": mean_shift_loss.item(),
                        "variance_shift_loss": variance_shift_loss.item(),
                        "trend_shift_loss": trend_shift_loss.item(),
                        "anomaly_loss": anomaly_loss.item(),
                        "current_lr": scheduler.get_last_lr()[0],
                    })
                iterations += 1
                
    artifact = wandb.Artifact(f"best_model_{wandb.run.id}", type='model')
    artifact.add_file(local_path = BEST_MODEL_PATH)
    artifact.save()
