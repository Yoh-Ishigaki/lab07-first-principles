import pytest
import torch
from model import BikeNet


@pytest.mark.parametrize("hidden_sizes", [(16, 8), (32, 16), (64, 32)])
def test_network_output_shape(hidden_sizes):
    model = BikeNet(input_size=10, hidden_sizes=hidden_sizes)
    batch_size = 8
    dummy_input = torch.randn(batch_size, 10)
    output = model(dummy_input)

    # Match the actual output shape returned by BikeNet
    assert output.shape == (batch_size,) or output.shape == (batch_size, 1)