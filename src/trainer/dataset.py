
import numpy as np
from torch.utils.data import Dataset
import torch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, str):
            self.X = np.load(X)
        else:
            self.X = X

        if isinstance(y, str):
            self.y = np.load(y)
        else:
            self.y = y
    
    def get_input_shape(self):
        return self.X.shape[1:]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]
        

