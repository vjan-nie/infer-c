# ADR-007 — Testable Predicates Generalize Beyond Matrix-to-Matrix Checks

## Status
Accepted

## Context

ADR-005 established a pattern: any dimension check that an `assert` would
otherwise enforce is factored into its own small, pure, `int`-returning
predicate function, so automated tests can exercise both the valid and
invalid cases without aborting the test binary. ADR-005's two original
examples — `mat_dims_compatible_for_mul`, `mat_dims_compatible_for_add_bias`
— both take two `const Matrix*` arguments and validate a relationship
between two already-constructed matrices.

M0b needed to guard `mat_create(int rows, int cols)` against invalid
(non-positive) dimensions, *before* any `Matrix` exists. This is a
precondition on raw input, not a relationship between two matrices — a
narrower reading of ADR-005 could argue the pattern doesn't apply here.

## Decision

The pattern applies. ADR-005's actual rationale for testable predicates —
"an `assert` inside a function can't be exercised by an automated test
without aborting the whole suite, so factor the check into something a
test can call directly" — has nothing to do with the check happening to
compare two matrices. It applies equally to validating raw scalar
preconditions before any structure is built.

`mat_dims_valid(int rows, int cols)` was added on this basis: same shape of
problem (an `assert`-guarded precondition that needs independent test
coverage), different shape of inputs (`int`s, not `Matrix*`s).

The general rule, stated explicitly so it doesn't need rediscovering at
each module:

> Any precondition enforced by `assert` — regardless of whether it
> compares two existing structures or validates raw input before a
> structure exists — gets factored into its own pure, testable predicate
> if there is any value in covering both its valid and invalid paths with
> an automated test.

## Consequences

- Future modules (`nn.c` in M2, validating e.g. that a layer's declared
  dimensions are positive, or that consecutive layers' dimensions are
  chain-compatible) should default to this same pattern rather than
  treating M0's two matrix-to-matrix predicates as the only sanctioned
  shape.
- Not every `assert` needs a predicate — only ones where the invalid path
  is worth covering by a test. A predicate that would only ever be called
  with one trivially-always-true input isn't worth extracting; this ADR
  generalizes the pattern, it doesn't mandate maximal extraction.
- This ADR does not change ADR-005's content; it clarifies scope. ADR-005
  remains the source of the original decision and its two original
  examples.
