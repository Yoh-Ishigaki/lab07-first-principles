
"""Data splitting, preprocessing, Dataset and DataLoaders."""

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC_FEATURES = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]

CATEGORICAL_FEATURES = ["station_type"]


class BikeDataset(Dataset):
    """Custom Dataset returning one row at a time."""

    def __init__(self, features, targets):
        self.features = torch.tensor(
            np.asarray(features, dtype=np.float32),
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            np.asarray(targets, dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def split_data(frame, seed=42):
    """Split data into 60% training, 20% validation and 20% testing."""
    train_frame, temporary = train_test_split(
        frame,
        test_size=0.40,
        random_state=seed,
    )

    val_frame, test_frame = train_test_split(
        temporary,
        test_size=0.50,
        random_state=seed,
    )

    return (
        train_frame.reset_index(drop=True),
        val_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def prepare_data(frame=None, seed=42):
    """Split data and fit preprocessing using training rows only."""
    if frame is None:
        frame = make_rentals()

    train_frame, val_frame, test_frame = split_data(frame, seed)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(
        train_frame.drop(columns=[TARGET])
    )
    X_val = preprocessor.transform(
        val_frame.drop(columns=[TARGET])
    )
    X_test = preprocessor.transform(
        test_frame.drop(columns=[TARGET])
    )

    y_train = train_frame[TARGET].to_numpy(dtype=np.float32)
    y_val = val_frame[TARGET].to_numpy(dtype=np.float32)
    y_test = test_frame[TARGET].to_numpy(dtype=np.float32)

    train_dataset = BikeDataset(X_train, y_train)
    val_dataset = BikeDataset(X_val, y_val)
    test_dataset = BikeDataset(X_test, y_test)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        preprocessor,
    )


def make_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=64,
    seed=42,
):
    """Create DataLoaders, shuffling training batches only."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
