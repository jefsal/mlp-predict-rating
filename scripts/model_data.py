"""Prepare normalized tensors and DataLoaders for the MLP."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from split_data import load_forecasts, split_by_date

SUMMARY_PATH = Path(__file__).resolve().parents[1] / "data/processed/summary.json"
TARGET = "target_rating"
BATCH_SIZE = 32


def load_feature_names(path=SUMMARY_PATH):
    """The official list of model inputs, taken from summary.json."""
    with open(path) as file:
        return json.load(file)["features"]


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


def prepare_data():
    """Return normalized (X, y) tensors for each split, plus the training statistics.

    The mean and std are computed on the training rows only and then applied to
    validation and test, so nothing from the later matches leaks into training.
    """
    features = load_feature_names()
    train, validation, test = split_by_date(load_forecasts())

    stats = {
        "x_mean": train[features].mean(),
        "x_std": train[features].std(),
        "y_mean": train[TARGET].mean(),
        "y_std": train[TARGET].std(),
    }

    splits = {}
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        X = (part[features] - stats["x_mean"]) / stats["x_std"]
        y = (part[TARGET] - stats["y_mean"]) / stats["y_std"]
        splits[name] = (
            torch.tensor(X.values, dtype=torch.float32),
            torch.tensor(y.values, dtype=torch.float32).view(-1, 1),
        )
    return splits, stats


def make_loaders(splits, batch_size=BATCH_SIZE):
    """One DataLoader per split; only the training loader shuffles."""
    return {
        name: DataLoader(
            MyDataset(X, y), batch_size=batch_size, shuffle=(name == "train")
        )
        for name, (X, y) in splits.items()
    }


if __name__ == "__main__":
    splits, stats = prepare_data()
    for name, (X, y) in splits.items():
        print(f"{name:10s} X {tuple(X.shape)}  y {tuple(y.shape)}")

    X_train, y_train = splits["train"]
    print("train X mean (should be ~0):", X_train.mean(dim=0).abs().max().item())
    print("train X std  (should be ~1):", X_train.std(dim=0).mean().item())
    print("train y mean/std:", y_train.mean().item(), y_train.std().item())

    X_val, _ = splits["validation"]
    print("validation X mean (close to, but not exactly, 0):", X_val.mean().item())

    loaders = make_loaders(splits)
    features, labels = next(iter(loaders["train"]))
    print("one batch:", tuple(features.shape), tuple(labels.shape))  # (32, 12) (32, 1)
