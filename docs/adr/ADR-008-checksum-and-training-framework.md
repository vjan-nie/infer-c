# ADR-008 — Checksum Algorithm and Training Framework

## Status
Accepted

## Context

Two decisions were left open going into M1:

- ADR-004 left the weight-file checksum algorithm unfixed ("simple sum is
  enough... CRC32 is more robust... pending closure when train.py and
  model.c are written").
- The M0/M0b scaffold did not commit to any particular Python ML stack for
  `train.py`, since training was explicitly out of scope until M1.

## Decision 1 — Checksum: CRC32

CRC32 is used over the weight file's content. (The exact byte range —
the entire file including `magic`/`format_version`, except the trailing
checksum field itself — is fixed precisely in ADR-010, which also fixes
this decision's other open point, byte order. This ADR's own original
wording understated that scope; ADR-010 is authoritative.)

Rationale: the implementation cost over a simple summed checksum is
marginal — Python's standard library (`zlib.crc32`) and a from-scratch C
implementation (a 256-entry lookup table, or a direct bit-by-bit
computation) are both small, well-understood pieces of code. In exchange,
CRC32 catches the overwhelming majority of realistic corruption patterns
(truncation, single/multi-bit flips, byte reordering) that a simple
additive checksum would miss. It is also immediately recognizable as "the
correct way to do this" to anyone reviewing the project, which matters for
a portfolio piece. No new dependency is introduced on either side: `zlib`
is Python stdlib, and the C side implements CRC32 itself (consistent with
the project's "from scratch, no frameworks" posture — CRC32 is a checksum
algorithm, not an ML framework, so implementing it directly is in scope).

## Decision 2 — Training framework: PyTorch

`train.py` uses PyTorch (plus NumPy for any array manipulation outside the
training loop itself) to define and train the MLP.

Rationale: PyTorch exposes trained weights as directly inspectable tensors
(`model.fc1.weight.data.numpy()`), which keeps the export step — converting
trained weights into this project's binary format — transparent and easy
to explain. This matters for a project whose value proposition rests on
understanding every piece of the pipeline, not on treating training as a
black box that "just produces a file." Keras/TensorFlow's higher-level
APIs tend to abstract this access more, which doesn't serve that goal as
well, even though either framework would produce a numerically equivalent
model.

This choice is confined to `train.py`. It has no bearing on `infer-c`'s own
"no ML frameworks" rule (ADR-001's independence and the project's core
premise) — that rule applies to the C inference engine, not to the
separate, explicitly-out-of-scope Python training step.

## Consequences

- `model.c`'s loader must implement CRC32 verification matching exactly
  what `train.py` computes — same byte range, same polynomial/algorithm
  variant (standard CRC-32, as used by zlib/gzip/PNG: polynomial
  0xEDB88320, initial value 0xFFFFFFFF, final XOR 0xFFFFFFFF). This exact
  parameterization must be documented in both `train.py` and `model.c` so
  neither side can drift from the other silently.
- `train.py`'s dependencies become PyTorch + NumPy (+ whatever MNIST
  loading convenience PyTorch's `torchvision` offers, or a manual
  IDX-format download/parse — to be decided when writing `train.py`
  itself, not part of this ADR).
- This ADR closes the two items ADR-004 left pending. ADR-004's content is
  not edited; this is a follow-up decision, consistent with how this
  project's ADRs accumulate rather than get rewritten in place.
