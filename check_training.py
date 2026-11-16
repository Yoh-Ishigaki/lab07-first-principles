
"""Check the complete training process."""

import torch
from torch.optim import Adam

from model import BikeNet
from prepare import make_loaders, prepare_data
from train import train_model


def main():
    """Run a short training test."""
    torch.manual_seed(42)

    train_data, val_data, test_data, _ = prepare_data(seed=42)

    train_loader, val_loader, _ = make_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=64,
        seed=42,
    )

    input_size = train_data[0][0].shape[0]
    model = BikeNet(input_size=input_size)

    optimizer = Adam(model.parameters(), lr=0.001)

    history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=20,
    )

    print("\nTraining test summary")
    print("---------------------")
    print(f"Total epochs: {len(history)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation MAE: {min(row['val_mae'] for row in history):.4f}")


if __name__ == "__main__":
    main()
