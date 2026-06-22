"""
export_weights.py — Exports a trained MLP's weights to infer-c's
binary weight format (ADR-004), with a CRC32 checksum (ADR-008,
ADR-010).

Format (all multi-byte fields little-endian, per ADR-010):
  [magic: "IC01"][format_version: u32][n_layers: u32]
  per layer:
    [activation_id: u32]  (0=none, 1=relu, 2=softmax)
    [rows: u32][cols: u32]      -- rows=out_features, cols=in_features
    [weights: rows*cols float32, row-major]
    [bias: rows float32]
  [crc32: u32]  -- standard CRC-32 (zlib), over magic + everything up
                   to (not including) the checksum field itself
"""

import struct
import zlib

MAGIC = b"IC01"
FORMAT_VERSION = 1

ACTIVATION_NONE = 0
ACTIVATION_RELU = 1
ACTIVATION_SOFTMAX = 2


def export_weights(model, path):
    """
    model: the trained MLP (model.py's MLP class).
    path: output file path for the binary weight file.
    """
    layers = [
        (model.fc1.weight, model.fc1.bias, ACTIVATION_RELU),
        (model.fc2.weight, model.fc2.bias, ACTIVATION_SOFTMAX),
    ]

    # Everything that gets checksummed: magic + version + n_layers + all
    # layer data (ADR-010 — checksum covers the header too).
    content = MAGIC
    content += struct.pack("<I", FORMAT_VERSION)
    content += struct.pack("<I", len(layers))

    for weight, bias, activation_id in layers:
        w = weight.detach().numpy()  # shape (out_features, in_features)
        b = bias.detach().numpy()    # shape (out_features,)
        rows, cols = w.shape

        content += struct.pack("<I", activation_id)
        content += struct.pack("<II", rows, cols)
        content += w.astype("<f4").tobytes()  # row-major, little-endian
        content += b.astype("<f4").tobytes()

    checksum = zlib.crc32(content) & 0xFFFFFFFF

    with open(path, "wb") as f:
        f.write(content)
        f.write(struct.pack("<I", checksum))
