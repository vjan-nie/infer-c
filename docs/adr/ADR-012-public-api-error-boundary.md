# ADR-012 — Error Handling at the Public API Boundary

## Status
Accepted

## Context

M2.5's audit needed to decide how `infer_run` (the public API's
inference entry point) should react to a caller-supplied `input_size`
that doesn't match the loaded model's expected input width, or a `NULL`
`InferModel*`.

ADR-005 established `assert` for programmer errors and `NULL`/error
signaling for environment errors, for this project's *internal* call
sites (e.g. `mat_mul`'s dimension checks, where every caller is code
within this same codebase). ADR-011 further established that values
derived from file content remain environment-error category regardless
of how many earlier checks already passed.

Neither ADR directly answers the question M2.5 faced: a wrong
`input_size` passed to `infer_run` is a programmer error in the sense
that it reflects a mistake in calling code, not file corruption — but
that calling code is, at the public API boundary, code this project
does not own and cannot fix or verify by testing it. ADR-005's
justification for using `assert` ("the calling code is correct, because
this project controls and tests it") does not hold here.

## Decision

**At the public API boundary (`infer_c.h`), no caller mistake — however
clearly a "programmer error" in spirit — is ever handled with `assert`
or any other process-aborting behavior.** Internal call sites (anything
not reachable directly by an external consumer of this library) continue
to follow ADR-005's original assert/NULL split unchanged.

This is a third category, distinct from both of ADR-005's:

| | Caller is... | Reaction |
|---|---|---|
| ADR-005 programmer error | This codebase's own code | `assert` |
| ADR-005/011 environment error | File content, OS, allocator | `NULL` / sentinel |
| **ADR-012 public-boundary error** | **External code linking this library** | **Sentinel result, never abort** |

Rationale: an `assert`-triggered abort inside a library terminates the
*entire host process* that linked against it — not just this project's
own test binary. The blast radius of a public library crashing its
caller over a parameter mistake is categorically larger than an internal
assertion failing during this project's own development and testing.
A caller passing a wrong `input_size` (an off-by-one, a stale constant,
a copy-paste mistake) is exactly the kind of error a real external
integrator will make at some point — handling it by taking down their
process is hostile API design, not "correctly enforcing a contract."

Concretely, for `infer_run`: a mismatched `input_size`, or a `NULL`
`InferModel*`, produces a sentinel `InferResult` —
`{ .predicted_class = -1, .confidence = 0.0f }` — documented in
`infer_c.h` purely in terms of this API's own vocabulary, never
referencing internal types (`Matrix`, `Layer`) even informally in
comments. `-1` is an unambiguous out-of-range sentinel since
`predicted_class` is otherwise always in `[0, n_classes)`.

## Consequences

- `infer_c.h`'s documentation must state this sentinel behavior
  explicitly as part of the function's contract, not as an
  implementation detail — a caller checking `predicted_class >= 0`
  before trusting a result is now part of correctly using this API.
- This pattern (sentinel result, never abort, at any public-facing
  entry point) should be the default for any future public API addition
  to `infer_c.h`. If a future public function's only way to signal an
  error doesn't fit a sentinel value cleanly, that's a signal to
  reconsider `InferResult`'s shape deliberately (e.g. an explicit
  success/error field) rather than reaching for `assert` out of
  convenience.
- Internal modules (`matrix.c`, `model.c`, `nn.c`) are unaffected — this
  ADR does not relax or revisit any existing `assert` in code that is
  never called directly by an external consumer.
