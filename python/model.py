"""
model.py — Defines the MLP architecture (784 -> 128 -> 10) used by
train.py. Matches the inference engine's expected layer structure
(ADR-004, ADR-009).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x  # raw logits — softmax applied outside the model
