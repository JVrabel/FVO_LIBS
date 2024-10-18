# LIBS Data Classification

This project implements a simple Multi-Layer Perceptron (MLP) neural network to classify LIBS (Laser-Induced Breakdown Spectroscopy) data from the EMSLIBS contest 2019 benchmark classification dataset.

## Project Structure

- `src/`: Contains all source code files
  - `train.py`: Main script for training and testing the model
  - `model_builder.py`: Defines the MLP model architecture
  - `run_experiment.py`: Script to run experiments with different parameters
  - `load_libs_data.py`: Functions to load LIBS data from H5 files
  - `data_setup.py`: Prepares data loaders for training and testing
  - `utils.py`: Utility functions for saving results
  - `engine.py`: Contains training and testing loop implementations
- `data/`: Directory for storing input data files (not included in the repository)
- `models/`: Directory for saving trained models
- `output/`: Directory for storing experiment results and logs

## Setup

1. Ensure you have Python 3.x installed.
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model with default parameters:

```
python src/train.py
```

You can customize the training process using command-line arguments:

```
python src/train.py --epochs 100 --batch_size 64 --learning_rate 0.0005 --save_model
```

### Running Experiments

To run experiments with different hyperparameters:

```
python src/run_experiment.py
```

You can modify the `run_experiment.py` file to set up different experimental configurations.

### Testing the Model

To test a pre-trained model:

```
python src/train.py --test-only
```

## Data

This project uses the EMSLIBS contest 2019 benchmark classification dataset. The data files are not included in this repository due to size constraints. Please place the following files in the `data/` directory:

- `contest_TRAIN.h5`: Training dataset
- `contest_TEST.h5`: Test dataset
- `test_labels.csv`: Labels for the test dataset

## Model Architecture

The project uses a simple Multi-Layer Perceptron (MLP) with two hidden layers. The architecture is defined in `src/model_builder.py`.

## Results

After training and testing, the results will be saved in the `output/` directory. Each run creates a timestamped subdirectory containing:

- `test_results.txt`: Detailed classification report and confusion matrix
- `test_predictions.npy`: NumPy array of model predictions
- `test_true_labels.npy`: NumPy array of true labels

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- EMSLIBS contest 2019 for providing the benchmark dataset
- PyTorch community for the excellent deep learning framework
