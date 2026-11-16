import torch
from model import BikeNet
from train import run_epoch


def test_optimizer_step_changes_weights(prepared_splits):
    train_ds = prepared_splits["train_ds"]
    model = BikeNet(input_size=train_ds.features.shape[1], hidden_sizes=(32, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    initial_weights = [p.clone() for p in model.parameters()]

    loader = prepared_splits["train_loader"]
    x, y = next(iter(loader))
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y.unsqueeze(1))
    loss.backward()
    optimizer.step()

    for init_w, curr_w in zip(initial_weights, model.parameters()):
        assert not torch.allclose(init_w, curr_w)


def test_memorize_small_batch(prepared_splits):
    train_ds = prepared_splits["train_ds"]
    model = BikeNet(input_size=train_ds.features.shape[1], hidden_sizes=(32, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.MSELoss()

    loader = prepared_splits["train_loader"]
    x, y = next(iter(loader))
    
    # Ensure target shape matches prediction shape
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)

    initial_loss = None
    for _ in range(300):
        optimizer.zero_grad()
        pred = model(x)
        if pred.ndim > 1 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
            
        loss = criterion(pred, y)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1 * initial_loss


def test_validation_leaves_weights_unchanged(prepared_splits):
    train_ds = prepared_splits["train_ds"]
    model = BikeNet(input_size=train_ds.features.shape[1], hidden_sizes=(32, 16))
    val_loader = prepared_splits["val_loader"]
    criterion = torch.nn.MSELoss()

    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    run_epoch(model, val_loader, criterion, optimizer=None)
    final_state = {k: v.clone() for k, v in model.state_dict().items()}

    for k in initial_state:
        assert torch.allclose(initial_state[k], final_state[k])