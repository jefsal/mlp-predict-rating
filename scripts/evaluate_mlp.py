"""Evaluate the MLP against simple baselines and plot the training curves."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from model_data import TARGET, make_loaders, prepare_data
from split_data import load_forecasts, split_by_date
from train_mlp import MLP, train


def predict(model, loader):
    """Predictions for every row of a loader, in the loader's order (standardized units)."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for features, _ in loader:
            outputs.append(model(features))
    return torch.cat(outputs).squeeze(1).numpy()


def mae_rmse(y_true, y_pred):
    """Mean absolute error and root mean squared error, in rating points."""
    error = y_pred - y_true
    return np.abs(error).mean(), np.sqrt((error ** 2).mean())


def evaluate_validation(model, loaders, stats):
    """Print MAE and RMSE on the validation set for the MLP and the two baselines."""
    validation = split_by_date(load_forecasts())[1]
    y_true = validation[TARGET].to_numpy()

    # The validation loader does not shuffle, so its rows match this table's order.
    # The model works in standardized units, so convert back to rating points.
    mlp_pred = predict(model, loaders["validation"]) * stats["y_std"] + stats["y_mean"]
    mean_pred = np.full_like(y_true, stats["y_mean"])  # training mean for every row
    recent_pred = validation["past_5_rating_mean"].to_numpy()

    results = {
        "Always the training mean": mean_pred,
        "Recent average (last 5)": recent_pred,
        "MLP": mlp_pred,
    }
    print(f"{'Validation':26s} {'MAE':>6s} {'RMSE':>6s}")
    for name, pred in results.items():
        mae, rmse = mae_rmse(y_true, pred)
        print(f"{name:26s} {mae:6.3f} {rmse:6.3f}")


def plot_losses(train_losses, validation_losses, path=None):
    """Training vs. validation loss by epoch. Save to `path` if given, then show."""
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(epochs, validation_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (standardized units)")
    plt.title("Training vs. validation loss")
    plt.legend()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(123)

    splits, stats = prepare_data()
    loaders = make_loaders(splits)

    model = MLP(num_features=splits["train"][0].shape[1])
    train_losses, validation_losses = train(model, loaders, num_epochs=100, lr=0.01)

    last10 = sum(validation_losses[-10:]) / 10
    print(f"Average validation MSE over the last 10 epochs: {last10:.4f}")

    evaluate_validation(model, loaders, stats)
    plot_losses(train_losses, validation_losses, path="loss_curves.png")