import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def save_results(run_dir, accuracy, true_labels, test_predictions):
    with open(os.path.join(run_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true_labels, test_predictions, target_names=[f"Class {i}" for i in range(1, 13)]))
        f.write("\nConfusion Matrix:\n")
        cm = confusion_matrix(true_labels, test_predictions)
        f.write(str(cm))
    
    print(f"Detailed results saved to {os.path.join(run_dir, 'test_results.txt')}")
    
    # Convert 0-indexed predictions back to 1-indexed
    test_predictions = test_predictions + 1
    true_labels = true_labels + 1
    
    np.save(os.path.join(run_dir, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(run_dir, 'test_true_labels.npy'), true_labels)
    print(f"Test predictions and true labels saved in {run_dir}")
