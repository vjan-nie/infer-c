"""
export_verification_samples.py — Exports a fixed set of test samples,
with the trained model's exact output, for cross-checking the C
forward pass against this Python reference in M2 (ADR-009).

Format (little-endian, per ADR-010's convention):
  [magic: "ICV1"][n_samples: u32][input_size: u32][n_classes: u32]
  per sample:
    [input: input_size float32]
    [expected_logits: n_classes float32]
    [expected_probs: n_classes float32]
    [expected_class: u32]
"""

import struct
import torch
import torch.nn.functional as F

MAGIC = b"ICV1"


def export_verification_samples(model, x_samples, path, n_samples=20):
    """
    model: the trained MLP, already in eval mode by the caller if desired.
    x_samples: tensor of shape (N, input_size) — typically a slice of
               the test set, already normalized the same way training
               data was (see train.py's load_dataset).
    path: output file path.
    n_samples: how many samples (from the start of x_samples) to export.
    """
    model.eval()
    with torch.no_grad():
        x = x_samples[:n_samples]
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        predicted = torch.argmax(logits, dim=1)

    input_size = x.shape[1]
    n_classes = logits.shape[1]

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", n_samples, input_size, n_classes))

        for i in range(n_samples):
            f.write(x[i].numpy().astype("<f4").tobytes())
            f.write(logits[i].numpy().astype("<f4").tobytes())
            f.write(probs[i].numpy().astype("<f4").tobytes())
            f.write(struct.pack("<I", int(predicted[i].item())))
