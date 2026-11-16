import pytest
from bikes_data import make_rentals
from prepare import make_loaders, prepare_data


@pytest.fixture(scope="session")
def raw_data():
    return make_rentals()


@pytest.fixture(scope="session")
def prepared_splits(raw_data):
    train_ds, val_ds, test_ds, preprocessor = prepare_data(raw_data)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "preprocessor": preprocessor,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }