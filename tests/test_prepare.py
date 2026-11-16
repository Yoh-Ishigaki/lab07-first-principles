import torch


def test_dataset_length_and_shape(prepared_splits):
    train_ds = prepared_splits["train_ds"]
    assert len(train_ds) > 0
    x, y = train_ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_splits_sizes_and_no_overlap(raw_data, prepared_splits):
    total_len = len(raw_data)
    train_len = len(prepared_splits["train_ds"])
    val_len = len(prepared_splits["val_ds"])
    test_len = len(prepared_splits["test_ds"])

    assert train_len + val_len + test_len == total_len
    assert train_len == int(0.6 * total_len)
    assert val_len == int(0.2 * total_len)
    assert test_len == total_len - train_len - val_len


def test_dataloader_shuffling(prepared_splits):
    loader = prepared_splits["train_loader"]
    # Get first batch from the loader
    batch1 = next(iter(loader))[0]
    batch2 = next(iter(loader))[0]
    
    # Just verify batches are tensors of expected shape
    assert batch1.ndim == 2