# ADR-004 — Weight File Format

## Status
Accepted

## Context

`infer-c` separates training (Python) from inference (C). This requires a
binary file contract that both sides honor: `train.py` writes it, `model.c`
reads it. The format must:

- Be parseable in pure C99, with no external libraries (consistent with the
  project's "from scratch" goal).
- Allow early, explicit detection of a mismatch between the file and the
  parser (wrong format version, corrupted file), instead of failing with a
  silent segfault or a read of garbage memory.
- Be extensible without a rewrite if new activation functions are added to
  later layers, without turning into a general-purpose ML format (ONNX)
  that exceeds the MVP's scope.

## Options considered

**A — Minimal flat binary.** Header with magic + layer count, then per
layer: dimensions, weights, bias. As small as possible. Rejected as the
final version (though it's the basis of the chosen option) because it does
not distinguish format versions or per-layer activation functions, which
makes it harder to diagnose failures and evolve the format later.

**B — JSON with base64/hex-encoded weights.** Readable and easy to generate
from Python. Rejected: parsing JSON in pure C99 without libraries is
non-trivial work that adds nothing to the project's goal (the value is in
the inference engine, not in reimplementing a text parser). Pulling in a
JSON library would break the project's dependency economy without real
need.

**C — Standard format (`.npy`/ONNX).** Genuinely interoperable with the
Python/ML ecosystem. Rejected for the MVP: `.npy` has a Python-dict-style
header that is non-trivial to parse in C without libraries; ONNX is
protobuf, oversized for a two-layer MLP. Kept as an explicit *stretch goal*
(already in the project's original scope), not as the MVP's starting point.

**D — Flat binary with extended self-describing header.** Like A, plus
format version, per-layer activation id, and a checksum. Chosen — see
Decision.

## Decision

Custom binary format, version 1:

```
[magic: 4 bytes, "IC01"]
[format_version: uint32]
[n_layers: uint32]
for each layer:
  [activation_id: uint32]   // 0=none, 1=relu, 2=softmax
  [rows: uint32][cols: uint32]
  [weights: rows*cols floats, row-major]
  [bias: rows floats]
[checksum: uint32]           // over the content above, algorithm to be fixed in M1
```

Explicit assumptions, not implicit ones:

- Same endianness on both sides (training and inference run on the same
  architecture in the MVP). No cross-endian portability is guaranteed.
- 32-bit `float` throughout the project (to be confirmed in M0 if not
  already fixed — must be consistent with `matrix.c`'s internal
  representation).

## Consequences

- `model.c` must validate `magic` and `format_version` before reading
  anything else, and fail explicitly (not silently) if they don't match.
- Adding a new activation function in the future does not break the
  format: it only adds a new `activation_id` value. Adding a different
  layer type (e.g. convolutional, stretch goal) will require a format
  revision and a `format_version` bump.
- `train.py` is the single source of truth for how the file is written;
  any format change happens there first and is documented in this ADR (as
  a new revision section, not by rewriting the original decision).
- The checksum requires picking a concrete algorithm in M1 (a simple sum
  is enough to catch basic truncation/corruption; CRC32 is more robust and
  not much more expensive to implement). Pending closure when `train.py`
  and `model.c` are written.
