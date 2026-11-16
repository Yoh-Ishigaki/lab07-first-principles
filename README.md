# Lab 07 – First Principles Neural Network

## 1. Project Overview

This project develops a feed-forward neural network using PyTorch to predict bike rental demand.

The project explores neural network training, learning rates, hidden layer configurations, and model evaluation.

## 2. Model Architecture

The neural network uses:
- Input layer based on the number of input features.
- Two hidden layers.
- ReLU activation functions.
- One output neuron for numerical prediction.

## 3. Experiments

### Main Training Run
- Learning rate: 0.001
- Hidden layer sizes: (64, 32)
- Training epochs: 100

### Experiment 1 – Learning Rate

- Learning rate: 0.0001
- Hidden layer sizes: (64, 32)
- Training epochs: 100

**Analysis:** The learning curve shows that training and validation losses decrease gradually. The model continues improving throughout the 100 epochs.

### Experiment 2 – Hidden Layer Sizes

- Learning rate: 0.001
- Hidden layer sizes: (32, 16)
- Training epochs: 100
- Best epoch: 100
- Best validation loss: 96.4987

**Analysis:** Both training and validation losses decrease consistently. There is no obvious indication of overfitting. The model demonstrates stable learning.

## 4. Experiment Comparison

The main training run currently demonstrates the lowest validation loss based on the available learning curves.

Experiment 1 uses a smaller learning rate, while Experiment 2 uses a smaller network architecture.

The original configuration appears to provide the strongest validation performance.

## 5. Final Model Selection

The main training configuration is the current leading candidate:

- Learning rate: 0.001
- Hidden layer sizes: (64, 32)

Final selection will be confirmed after test dataset evaluation.

## 6. Final Evaluation

To be completed after evaluating the selected model on the test dataset.

- Test MSE: Pending
- Test MAE: Pending

## 7. Conclusion

This project demonstrates how learning rates and neural network architecture influence model training and performance.

The experiments show the importance of comparing training and validation losses before selecting a final model.

The final model will be selected based on validation performance and evaluated using the held-out test dataset.
