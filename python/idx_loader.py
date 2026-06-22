"""
idx_loader.py — Parses MNIST IDX files (images and labels) using only
the Python standard library and NumPy. No torchvision, no dataset helpers.
"""

import struct
import numpy as np


def load_idx_images(path):
    """Returns a NumPy array of shape (n_images, rows, cols), dtype uint8."""
    with open(path, "rb") as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(
                f"{path}: unexpected magic number {magic:#x}, "
                f"expected 0x00000803 (image file)"
            )
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n_images, rows, cols)


def load_idx_labels(path):
    """Returns a NumPy array of shape (n_labels,), dtype uint8."""
    with open(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(
                f"{path}: unexpected magic number {magic:#x}, "
                f"expected 0x00000801 (label file)"
            )
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n_labels)
