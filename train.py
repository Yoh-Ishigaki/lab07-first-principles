"""Training and evaluation functions for bike rental prediction."""

import copy
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn

from model import BikeNet
from prepare import make_loaders, prepare_data


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """Run one training or evaluation epoch.

    If optimizer is provided, update model weights.
    Otherwise, evaluate without changing model weights.

    Returns:
        Tuple containing mean loss and mean absolute error.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions = model(features)
            loss = criterion(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = len(targets)

        total_loss += loss.item() * batch_size
        total_absolute_error += (
            torch.abs(predictions - targets).sum().item()
        )
        total_samples += batch_size

    mean_loss = total_loss / total_samples
    mean_mae = total_absolute_error / total_samples

    return mean_loss, mean_mae


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs=100,
    device="cpu",
):
    """Train a model and restore its best validation weights.

    Returns:
        A history list and the best epoch number.
    """
    criterion = nn.MSELoss()
    model.to(device)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_mae = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    print(f"Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")

    return history, best_epoch


def evaluate_baselines(
    train_dataset,
    val_dataset,
    test_dataset,
):
    """Evaluate mean prediction and linear regression baselines."""
    training_mean = float(np.mean(train_dataset.targets.numpy()))

    val_mean_predictions = np.full(len(val_dataset), training_mean)
    test_mean_predictions = np.full(len(test_dataset), training_mean)

    mean_val_mae = mean_absolute_error(
        val_dataset.targets.numpy(),
        val_mean_predictions,
    )
    mean_test_mae = mean_absolute_error(
        test_dataset.targets.numpy(),
        test_mean_predictions,
    )

    linear_model = LinearRegression()
    linear_model.fit(
        train_dataset.features.numpy(),
        train_dataset.targets.numpy(),
    )

    val_linear_predictions = linear_model.predict(
        val_dataset.features.numpy()
    )
    test_linear_predictions = linear_model.predict(
        test_dataset.features.numpy()
    )

    linear_val_mae = mean_absolute_error(
        val_dataset.targets.numpy(),
        val_linear_predictions,
    )
    linear_test_mae = mean_absolute_error(
        test_dataset.targets.numpy(),
        test_linear_predictions,
    )

    print("\n========== BASELINE RESULTS ==========")
    print(f"Mean Predictor Val MAE: {mean_val_mae:.4f}")
    print(f"Linear Regression Val MAE: {linear_val_mae:.4f}")

    return {
        "mean_val_mae": mean_val_mae,
        "mean_test_mae": mean_test_mae,
        "linear_val_mae": linear_val_mae,
        "linear_test_mae": linear_test_mae,
    }


def plot_learning_curves(
    history,
    best_epoch,
    filename="learning_curves.png"
):
    """Plot training and validation loss across epochs."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    plt.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        label=f"Best Epoch: {best_epoch}",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Neural Network Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(reports_dir / filename, dpi=300)
    plt.close()
    print(f"Learning curve saved to reports/{filename}")


if __name__ == "__main__":
    # Step 1: Prepare the datasets
    train_ds, val_ds, test_ds, preprocessor = prepare_data()

    # Step 2: Evaluate baseline models
    baseline_metrics = evaluate_baselines(train_ds, val_ds, test_ds)

    # Step 3: Create DataLoaders
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds
    )

    # Experiments setup list for experiments.csv
    experiments_data = []

    # --- MAIN RUN ---
    print("\n--- Running Main Model ---")
    model_main = BikeNet(
        input_size=train_ds.features.shape[1],
        hidden_sizes=(64, 32)
    )
    optimizer_main = torch.optim.Adam(model_main.parameters(), lr=0.001)
    history_main, best_epoch_main = train_model(
        model_main, train_loader, val_loader, optimizer_main, epochs=50
    )
    
    # Save main learning curve
    plot_learning_curves(
        history_main, best_epoch_main, filename="learning_curves.png"
    )

    # Evaluate main model on test set
    test_loss_main, test_mae_main = run_epoch(
        model_main, test_loader, nn.MSELoss(), optimizer=None
    )

    experiments_data.append({
        "run_name": "main_run",
        "best_epoch": best_epoch_main,
        "best_val_mae": history_main[best_epoch_main - 1]["val_mae"],
        "final_train_loss": history_main[-1]["train_loss"],
        "final_val_loss": history_main[-1]["val_loss"],
    })

    # --- EXPERIMENT 1: Higher Learning Rate ---
    print("\n--- Running Experiment 1 (High LR) ---")
    model_lr = BikeNet(
        input_size=train_ds.features.shape[1],
        hidden_sizes=(64, 32)
    )
    optimizer_lr = torch.optim.Adam(model_lr.parameters(), lr=0.05)
    history_lr, best_epoch_lr = train_model(
        model_lr, train_loader, val_loader, optimizer_lr, epochs=50
    )
    plot_learning_curves(
        history_lr, best_epoch_lr, filename="experiment_lr.png"
    )
    experiments_data.append({
        "run_name": "high_lr",
        "best_epoch": best_epoch_lr,
        "best_val_mae": history_lr[best_epoch_lr - 1]["val_mae"],
        "final_train_loss": history_lr[-1]["train_loss"],
        "final_val_loss": history_lr[-1]["val_loss"],
    })

    # --- EXPERIMENT 2: Hidden Layers Variant ---
    print("\n--- Running Experiment 2 (Larger Hidden Layers) ---")
    model_hidden = BikeNet(
        input_size=train_ds.features.shape[1],
        hidden_sizes=(128, 64, 32)
    )
    optimizer_hidden = torch.optim.Adam(model_hidden.parameters(), lr=0.001)
    history_hidden, best_epoch_hidden = train_model(
        model_hidden, train_loader, val_loader, optimizer_hidden, epochs=50
    )
    plot_learning_curves(
        history_hidden, best_epoch_hidden, filename="experiment_hidden.png"
    )
    experiments_data.append({
        "run_name": "larger_network",
        "best_epoch": best_epoch_hidden,
        "best_val_mae": history_hidden[best_epoch_hidden - 1]["val_mae"],
        "final_train_loss": history_hidden[-1]["train_loss"],
        "final_val_loss": history_hidden[-1]["val_loss"],
    })

    # --- SAVE REPORTS ---
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save history.csv for the main run
    history_df = pd.DataFrame(history_main)
    history_df.to_csv(reports_dir / "history.csv", index=False)

    # 2. Save experiments.csv comparing runs
    experiments_df = pd.DataFrame(experiments_data)
    experiments_df.to_csv(reports_dir / "experiments.csv", index=False)

    # 3. Save metrics.json for baselines and final test result
    metrics_dict = {
        "baselines": {
            "mean_predictor_val_mae": float(baseline_metrics["mean_val_mae"]),
            "linear_regression_val_mae": float(baseline_metrics["linear_val_mae"]),
        },
        "test_result": {
            "test_mae": float(test_mae_main),
            "test_loss": float(test_loss_main),
        },
    }
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print("\n========== FINAL TEST RESULTS (Main Model) ==========")
    print(f"Test MSE: {test_loss_main:.4f}")
    print(f"Test MAE: {test_mae_main:.4f}")
    print("\nAll reports successfully generated in reports/")