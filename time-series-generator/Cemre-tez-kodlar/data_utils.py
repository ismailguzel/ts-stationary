import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle 
import numpy as np

class DataFrameDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            self.dataframe_list = pickle.load(f)
        
    def __len__(self):
        return len(self.dataframe_list)

    def __getitem__(self, idx):
        sample = self.dataframe_list[idx]
        data = sample['data'].to_numpy()
        mean_shift = sample['mean_shift'].to_numpy().astype(np.float32)
        variance_shift = sample['variance_shift'].to_numpy().astype(np.float32)
        trend_shift = sample['trend_shift'].to_numpy().astype(np.float32)
        anomaly = sample['anomaly'].to_numpy().astype(np.float32)
        return data, mean_shift, variance_shift, trend_shift, anomaly

def create_loaders(filepaths, batch_size, shuffle = True):
    assert isinstance(filepaths, list)
    dsets = [DataFrameDataset(filepath) for filepath in filepaths]
    loaders = [DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last = True) for dset in dsets]
    
    return loaders
