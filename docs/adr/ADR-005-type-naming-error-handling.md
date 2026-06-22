# ADR-005 — Numeric Type, Naming Convention, and Error Handling

## Status
Accepted

## Context

The M0 audit (`01-audit.md`) surfaced three open questions blocking the
implementation of `matrix.h`: numeric type, function naming convention, and
how to handle invalid input (incompatible dimensions, allocation failure)
without sacrificing test coverage. These three decisions are foundational —
every later module (`nn.h`, `model.h`) inherits them.

## Decision 1 — Numeric type: `float`

The project uses `float` (32-bit) throughout, for weights, activations, and
all matrix/vector storage.

Rationale: MNIST does not require `double` precision; lightweight inference
engines (TFLite, ONNX Runtime) default to `float32`; and this is already the
implicit assumption in ADR-004 (weight file format). Using `double` would
double the size of the weight file and the memory footprint for no accuracy
benefit at this scale.

## Decision 2 — Naming convention: short prefix (`mat_`)

Functions use the short prefix `mat_` (`mat_create`, `mat_free`, `mat_mul`,
`mat_add_bias`, `mat_copy`), not the longer `matrix_` prefix the audit used
as a placeholder.

Rationale: these functions appear repeatedly, often several times per line,
in the forward-pass code that is the heart of this project. A shorter,
conventional prefix (in the spirit of BLAS-style naming) keeps that code
readable. This is a style decision more than a technical one, but it is
fixed now so `nn.h` and `model.h` inherit a consistent convention.

## Decision 3 — Error handling: dual convention, with a testable validation layer

Two classes of failure are treated differently, as the audit proposed:

- **Programmer errors** (incompatible dimensions passed to an operation):
  enforced with `assert()`. These should never happen if the calling code
  is correct — the model's architecture is fixed and known at load time.
- **Environment errors** (allocation failure): communicated via a `NULL`
  data pointer in the returned `Matrix`, which the caller must check.

This much matches the audit's proposal. The addition: rather than relying
purely on `assert()` for dimension checks — which cannot be exercised by an
automated test without aborting the whole test binary — each
dimension-compatibility check is factored into its own small, pure,
testable function:

```c
int mat_dims_compatible_for_mul(const Matrix* a, const Matrix* b);
int mat_dims_compatible_for_add_bias(const Matrix* m, const Matrix* bias);
```

These return `1`/`0` and never abort. `mat_mul` and `mat_add_bias` call the
corresponding function internally and `assert()` on its result. Tests call
the compatibility functions directly, with both valid and invalid inputs,
achieving full coverage of the validation logic without needing
`fork`/`waitpid` or any test infrastructure beyond a plain `assert`-based
runner.

This also pays forward into M2: when `nn.c` validates a loaded model's
layer dimensions against each other (a risk the audit flagged in its
"Risks" section), it can reuse these same compatibility functions to
validate the whole architecture *before* running a forward pass, instead of
discovering a mismatch mid-execution via an aborted `assert`.

## Consequences

- `matrix.h`'s public surface grows by two small predicate functions beyond
  what the audit sketched. This is a deliberate, minimal addition — not
  speculative configurability — justified directly by a concrete testing
  and validation need.
- `mat_mul` and `mat_add_bias` keep `assert`-based aborts for actual misuse;
  this ADR does not change that. It only makes the underlying check
  independently testable.
- `nn.c` (M2) should validate model architecture using the same
  `mat_dims_compatible_for_*` functions rather than inventing a parallel
  check.
- All weight files, in-memory matrices, and future Python exports
  (`train.py`) are `float32`. Any future move to `double` (unlikely, not
  currently justified) would require a new ADR and a weight-format version
  bump (see ADR-004).
