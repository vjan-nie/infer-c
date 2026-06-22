# ADR-009 — M1 Scope: Data Source, Architecture, Verification Export

## Status
Accepted

## Context

Three remaining open points before `train.py` can be written:

- How MNIST data is obtained and loaded.
- Final confirmation of the MLP architecture (already implicit in the
  project brief, formally confirmed here).
- Whether `train.py` should also export a small set of input/output pairs
  for cross-checking the C forward pass against the Python reference in
  M2, or whether that's a separate, later task.

## Decision 1 — Data source: manual IDX download + from-scratch parsing

`train.py` does not depend on `torchvision` (or any other dataset-helper
library) to obtain or load MNIST. Instead:

- The four standard MNIST IDX files (train/test images, train/test labels)
  are downloaded once via a small, separate, documented step (a short
  script or documented manual `curl`/`wget` commands — decided at
  implementation time, not part of this ADR).
- `train.py` parses the IDX format itself (a short, well-documented binary
  format: a fixed header with magic number, dimensions, followed by raw
  pixel/label bytes) using only the Python standard library (`struct`) and
  NumPy for array handling.

Rationale: avoids depending on `torchvision` purely for dataset
convenience, keeps the project reproducible without surprise network
calls hidden inside a library call, and — most importantly for this
project's learning goals — means the IDX format gets understood and
parsed once in Python before `data.c` has to parse the same format in
C (M3). Writing the parser twice, with the second time benefiting from
having already done it once, is exactly the kind of repetition that
builds real understanding rather than copying a format spec blind.

## Decision 2 — Architecture: confirmed 784 → 128 → 10

The MLP is exactly as specified in the original project brief: an input
layer of 784 units (28×28 flattened pixels), one hidden layer of 128
units with ReLU activation, and an output layer of 10 units (one per
digit class) with softmax. No changes. This is formally confirmed here so
it's a binding decision, not just an assumption inherited from the brief.

## Decision 3 — M1 also exports verification samples for M2

In addition to the trained weights (per ADR-004's format), `train.py`
exports a second, small file: a fixed number of test-set samples, each
with its input vector, the model's output (logits or post-softmax
probabilities — exact choice at implementation time, but must be stated
unambiguously in the file's own format), and the resulting predicted
class.

Rationale: M2's acceptance criterion is that the C forward pass matches
the Python reference within a small epsilon, on several samples. Producing
that reference data in M1 — while the trained PyTorch model is still
loaded and easy to query — is simpler and less error-prone than
reconstructing it later in a separate M2 task, which would otherwise need
to reload the trained model from the weight file and risk introducing a
second, slightly different code path for "run the Python reference
forward pass."

## Consequences

- `train.py`'s scope now includes: IDX parsing, model definition and
  training (PyTorch), weight export (ADR-004 format + CRC32 per ADR-008),
  and verification-sample export (this ADR). This is still one cohesive
  script appropriate to M1 — it is not scope creep into M2, since nothing
  about the *C* forward pass is implemented or assumed here.
- The verification-sample file's exact binary/text format is an
  implementation detail to fix when `train.py` is actually written (this
  ADR commits to producing it, not to its byte layout) — but it must be
  simple enough for a future C test harness (M2) to parse without needing
  anything beyond what `model.c`/`data.c` already know how to do (read
  floats and ints from a binary file).
- A held-out, fixed random seed should be used for any data shuffling or
  weight initialization in `train.py`, so that re-running training
  produces a deterministically comparable model — relevant for being able
  to regenerate the weight file and verification samples consistently if
  needed.
