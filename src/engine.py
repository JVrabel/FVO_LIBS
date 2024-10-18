import torch
import numpy as np
from sklearn.metrics import accuracy_score
import time

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            test_pred_logits = model(X)
            predictions.extend(test_pred_logits.argmax(dim=1).cpu().numpy())
            true_labels.extend(y.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device, run_dir):
    best_val_acc = 0
    best_model_state = None
    start_time = time.time()
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_predictions, val_true = test_step(model, val_dataloader, device)
        val_acc = accuracy_score(val_true, val_predictions)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch: {epoch+1}/{epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"val_acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model with validation accuracy: {best_val_acc:.4f}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    if best_model_state is not None:
        torch.save(best_model_state, f"{run_dir}/best_model.pth")
        print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")
    
    model.load_state_dict(best_model_state)
    return model
