# ADR-010 — Weight File Byte Order and Checksum Scope

## Status
Accepted

## Context

ADR-004 fixed the weight file's layout but left two implementation
details unresolved, surfaced while writing `export_weights.py`:

- Byte order (endianness) for all multi-byte numeric fields
  (`format_version`, `n_layers`, `rows`, `cols`, the `float32` weights and
  biases, and the checksum itself). ADR-004 only stated that the same
  architecture is assumed on both sides — it did not fix which order to
  actually use.
- The exact byte range the CRC32 checksum covers — whether it includes the
  `magic`/`format_version` header or only the layer data that follows.

## Decision 1 — Byte order: little-endian

All multi-byte fields in the weight file are little-endian.

Rationale: the project's target machines (x86/ARM, as used for both
training and inference in this project's scope) are natively
little-endian. Writing and reading little-endian values requires no byte
-swapping on either side — `struct.pack("<I", ...)` in Python and a plain
`fread` into a native `uint32_t`/`float` in C both do the right thing
without extra code. This trades cross-architecture portability (a
big-endian machine would read this file incorrectly without an explicit
swap) for simplicity on the architectures this project actually targets —
consistent with ADR-004's existing assumption that both sides run on the
same architecture.

Note this is the opposite convention from the MNIST IDX format itself
(big-endian, per the original NIST/LeCun specification) — these are two
unrelated formats with no obligation to match each other's byte order.

## Decision 2 — Checksum scope: entire file except the checksum field itself

The CRC32 checksum is computed over `magic + format_version + n_layers +`
all per-layer data — i.e., everything in the file except the trailing
4-byte checksum field.

Rationale: a checksum that only covers the layer data (excluding
`magic`/`format_version`) would fail to detect corruption in the header
itself — e.g. a flipped bit in `format_version` could cause `model.c` to
misinterpret the rest of the file while still passing a checksum that
never looked at that field. Covering the full file (minus the checksum
field, which obviously cannot include itself) catches corruption anywhere,
at the cost of computing CRC32 over a few extra bytes — a negligible cost
for a model this size.

## Decision 3 — Loader implication (binding on `model.c`, M2)

`model.c` must, when verifying a weight file:

1. Read the entire file.
2. Compute CRC32 over all bytes except the last 4 (the checksum field).
3. Compare against the last 4 bytes, read as a little-endian `uint32_t`.
4. Only then proceed to interpret `magic`, `format_version`, and the rest
   of the structure — checksum verification happens first, on the raw
   byte range, before any structured parsing.

This ordering matters: if checksum verification happened after parsing
individual fields, a corrupted `format_version` could already have sent
the loader down the wrong code path before the checksum check ever ran.

## Consequences

- `export_weights.py` (Python) and `model.c`'s loader (C, M2) must agree
  exactly on this byte order and checksum range. Any future change to
  either requires a `format_version` bump (per ADR-004) and a
  corresponding update to this ADR, not a silent edit.
- This ADR does not revise ADR-004's overall layout — it only fixes two
  details ADR-004 explicitly left open.
