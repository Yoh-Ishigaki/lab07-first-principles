
"""Neural network model for bike rental prediction."""

import torch
from torch import nn


class BikeNet(nn.Module):
    """Feed-forward neural network for bike rental regression."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...] = (32, 16),
    ):
        super().__init__()

        if not hidden_sizes:
            raise ValueError("At least one hidden layer is required.")

        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        # Output layer: one numerical prediction, without activation.
        layers.append(nn.Linear(previous_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one rental prediction per input row."""
        return self.network(x).squeeze(-1)
