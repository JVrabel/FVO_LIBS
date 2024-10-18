import subprocess
import sys

def run_experiment(test_only=False, epochs=300, batch_size=128, learning_rate=0.001,
                   reg_lambda=None, reg_type=None, save_model=False):
    command = [sys.executable, "src/train.py"]
    
    if test_only:
        command.append("--test-only")
    
    command.extend([
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate)
    ])
    
    if reg_lambda is not None:
        command.extend(["--reg_lambda", str(reg_lambda)])
    
    if reg_type is not None:
        command.extend(["--reg_type", reg_type])
    
    if save_model:
        command.append("--save_model")
    
    subprocess.run(command)

if __name__ == "__main__":
    # Example usage:
    run_experiment(
        epochs=50,
        batch_size=128,
        learning_rate=0.0005,
        reg_lambda=0.001,
        reg_type='l2',
        save_model=True
    )

    # Test only mode:
    # run_experiment(test_only=True)