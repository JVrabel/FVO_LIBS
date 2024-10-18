import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from load_libs_data import load_contest_train_dataset, load_contest_test_dataset

def create_dataloaders(train_file, batch_size, device, split_rate=0.3):
    X, y, _ = load_contest_train_dataset(train_file)
    print(f"Loaded X shape: {X.shape}")
    print(f"Original labels range: {y.min()} to {y.max()}")
    
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    y = y - 1
    
    print(f"Adjusted labels range: {y.min()} to {y.max()}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_rate, random_state=42, stratify=y, shuffle=True)
    
    scaler = Normalizer(norm='max')
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    X_train = torch.from_numpy(X_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    y_val = torch.from_numpy(y_val).long().to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader

def create_test_dataloader(test_file, test_labels_file, batch_size, device):
    X_test = load_contest_test_dataset(test_file)
    y_test = pd.read_csv(test_labels_file, header=None).values.squeeze() - 1
    
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()
    
    scaler = Normalizer(norm='max')
    X_test = scaler.fit_transform(X_test)
    
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_dataloader
