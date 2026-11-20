"""Define and train the MLP that forecasts a player's next match rating."""

import torch
import torch.nn.functional as F

from model_data import make_loaders, prepare_data


class MLP(torch.nn.Module):
    def __init__(self, num_features, num_outputs=1, hidden1=64, hidden2=32):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, hidden1),
            torch.nn.Sigmoid(),

            # 2nd hidden layer
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.Sigmoid(),

            # output layer (no activation: the output is the predicted rating)
            torch.nn.Linear(hidden2, num_outputs),
        )

    def forward(self, x):
        pred = self.all_layers(x)
        return pred


def compute_loss(model, loader):
    """Mean squared error over every row of a loader (no gradients, no learning)."""
    model.eval()
    total_error = 0.0
    total_rows = 0
    with torch.inference_mode():
        for features, labels in loader:
            pred = model(features)
            total_error += F.mse_loss(pred, labels, reduction="sum").item()
            total_rows += labels.shape[0]
    return total_error / total_rows


def train(model, loaders, num_epochs=100, lr=0.01):
    """Train with SGD and return the training and validation loss for every epoch."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_error = 0.0
        epoch_rows = 0

        for features, labels in loaders["train"]:
            pred = model(features)
            loss = F.mse_loss(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # weight by batch size so the smaller last batch counts fairly
            epoch_error += loss.item() * labels.shape[0]
            epoch_rows += labels.shape[0]

        train_losses.append(epoch_error / epoch_rows)
        validation_losses.append(compute_loss(model, loaders["validation"]))

        print(f"Epoch {epoch + 1:03d}/{num_epochs:03d} | "
              f"Train MSE {train_losses[-1]:.4f} | Val MSE {validation_losses[-1]:.4f}")

    return train_losses, validation_losses


if __name__ == "__main__":
    torch.manual_seed(123)

    splits, stats = prepare_data()
    loaders = make_loaders(splits)

    num_features = splits["train"][0].shape[1]  # 12
    model = MLP(num_features=num_features)

    train_losses, validation_losses = train(model, loaders, num_epochs=100, lr=0.01)

    # The losses above are in standardized units. Multiplying the root of the MSE
    # by y_std converts it back to rating points, comparable to the baselines.
    val_rmse = (validation_losses[-1] ** 0.5) * stats["y_std"]
    print(f"Final validation RMSE: {val_rmse:.3f} rating points")
