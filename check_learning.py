
"""Check that the neural network can memorize one small batch."""

import torch
from torch import nn
from torch.optim import Adam

from model import BikeNet
from prepare import make_loaders, prepare_data


def main():
    """Train on one batch and verify that the loss decreases."""
    torch.manual_seed(42)

    train_data, val_data, test_data, _ = prepare_data(seed=42)

    train_loader, _, _ = make_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=64,
        seed=42,
    )

    features, targets = next(iter(train_loader))

    model = BikeNet(input_size=features.shape[1])

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    model.train()

    with torch.no_grad():
        initial_loss = criterion(model(features), targets).item()

    print(f"Initial loss: {initial_loss:.6f}")

    for step in range(500):
        optimizer.zero_grad()

        predictions = model(features)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(
                f"Step {step + 1}: "
                f"Loss = {loss.item():.6f}"
            )

    with torch.no_grad():
        final_loss = criterion(model(features), targets).item()

    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"Loss reduction: {initial_loss / max(final_loss, 1e-12):.2f}x")

    if final_loss < initial_loss * 0.1:
        print("SUCCESS: The network learned the small batch!")
    else:
        print("WARNING: The loss did not decrease enough.")


if __name__ == "__main__":
    main()
