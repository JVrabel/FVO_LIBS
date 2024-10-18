import os
import torch
import argparse
from datetime import datetime
from torch import nn
from data_setup import create_dataloaders, create_test_dataloader
from model_builder import SimpleMLP
from engine import train, test_step
from utils import save_results
from sklearn.metrics import accuracy_score

# Setup directories
import win32com.client
shell = win32com.client.Dispatch("WScript.Shell")
shortcut = shell.CreateShortCut('data/contest_TRAIN.h5.lnk')
train_dir = shortcut.Targetpath
train_file = os.path.join(os.path.dirname(train_dir), 'contest_TRAIN.h5')
test_file = os.path.join(os.path.dirname(train_dir), 'contest_TEST.h5')
test_labels_file = 'data/test_labels.csv'
model_save_dir = "models"

# Hyperparameters
INPUT_SHAPE = 40000
OUTPUT_SHAPE = 12
HIDDEN_UNITS1 = 128
HIDDEN_UNITS2 = 64

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Create time-stamped run directory
    run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    if not args.test_only:
        print("Creating dataloaders...")
        train_dataloader, val_dataloader = create_dataloaders(train_file, args.batch_size, device)
    
    print("Creating test dataloader...")
    test_dataloader = create_test_dataloader(test_file, test_labels_file, args.batch_size, device)
    
    print("Initializing model...")
    model = SimpleMLP(INPUT_SHAPE, HIDDEN_UNITS1, HIDDEN_UNITS2, OUTPUT_SHAPE).to(device)
    
    if not args.test_only:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        os.makedirs(model_save_dir, exist_ok=True)
        
        best_model = train(model, train_dataloader, val_dataloader, optimizer, loss_fn, args.epochs, device, run_dir)
        
        if args.save_model:
            torch.save(best_model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))
    else:
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(os.path.join(model_save_dir, "best_model.pth")))
        best_model = model
    
    print("Evaluating on test set...")
    test_predictions, true_labels = test_step(best_model, test_dataloader, device)
    
    # Print statistics
    accuracy = accuracy_score(true_labels, test_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    save_results(run_dir, accuracy, true_labels, test_predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test MLP model for LIBS data")
    parser.add_argument("--test-only", action="store_true", help="Run only the test pipeline")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--reg_lambda", type=float, help="Regularization lambda")
    parser.add_argument("--reg_type", choices=['l1', 'l2', 'sparseloc'], help="Regularization type")
    parser.add_argument("--save_model", action="store_true", help="Save the model after training")
    
    args = parser.parse_args()
    
    main(args)
