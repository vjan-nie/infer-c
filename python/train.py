"""
train.py — Trains the MLP (784 -> 128 -> 10) on MNIST and prepares it
for export. See model.py for the architecture, idx_loader.py for data
loading.
"""

import torch
import torch.nn as nn
import numpy as np

from idx_loader import load_idx_images, load_idx_labels
from model import MLP

SEED = 42  # fixed, per ADR-009 — reproducible training run

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_dataset(images_path, labels_path):
    images = load_idx_images(images_path).astype(np.float32) / 255.0
    images = images.reshape(images.shape[0], -1)  # flatten 28x28 -> 784
    labels = load_idx_labels(labels_path).astype(np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


def train(model, x_train, y_train, epochs=10, batch_size=64, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n_samples = x_train.shape[0]

    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        total_loss = 0.0

        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.shape[0]

        avg_loss = total_loss / n_samples
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")


def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_test).float().mean().item()
    model.train()
    return accuracy


if __name__ == "__main__":
    x_train, y_train = load_dataset(
        "../data/raw/train-images-idx3-ubyte",
        "../data/raw/train-labels-idx1-ubyte",
    )
    x_test, y_test = load_dataset(
        "../data/raw/t10k-images-idx3-ubyte",
        "../data/raw/t10k-labels-idx1-ubyte",
    )

    model = MLP()
    train(model, x_train, y_train, epochs=10)

    accuracy = evaluate(model, x_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    