# ADR-013 — Validating File Content Against Architectural Assumptions, Not Just Format Integrity

## Status
Accepted

## Context

M3's audit designed `data_load_images` to reject any IDX image file whose
declared `rows`/`cols` aren't exactly `28x28`, even though the IDX format
itself places no such constraint — `python/idx_loader.py` (M1) accepts
any dimensions the file declares, since at that point in the pipeline
nothing yet depends on a specific size.

This project's `infer_run` (the public API, M2.5) expects a flat
`float* input` of exactly 784 elements, because the trained model's
first layer (`784 -> 128`, ADR-009 Decision 2) has no other valid input
width. A `data_load_images` that faithfully transcribed a `32x32` or
`20x20` IDX file (both structurally valid IDX files) would hand
`main.c`/`benchmark.c` a buffer of the wrong size, which would then
either trip `infer_run`'s own `input_size` mismatch sentinel
(ADR-012 — a caller-supplied-input error, correctly handled, but
discovered late and reported with no context about *why* the sizes
disagree) or, if `main.c` miscounted and passed a wrong `input_size`
value confidently, something worse.

## Decision

`data_load_images` validates `rows == 28 && cols == 28` as part of
loading, rejecting (returning `NULL`) any file that doesn't match —
even though this is stricter than the IDX format itself requires, and
stricter than `idx_loader.py`'s own validation.

This generalizes ADR-011's principle (file-content-derived values are
environment errors, validated before trusted) to a case ADR-011 didn't
originally cover: rejecting content that is *internally well-formed by
its own format's rules* but *inconsistent with a fixed assumption this
specific project's architecture depends on*. The IDX format doesn't
care what `rows`/`cols` are; this project does, because the model it
loads was trained on, and only ever expects, `28x28` inputs.

## Consequences

- This is a project-specific constraint, not a general IDX-parsing
  rule. If this project's scope ever changed to support multiple model
  architectures with different input sizes, this hardcoded check would
  need to become a parameter rather than a constant — that's a real
  future cost of this decision, accepted here because the project's
  actual scope (one architecture, MNIST, 28x28) doesn't currently
  justify that flexibility (`critical-software-engineering`'s
  "no speculative configurability" principle).
- This failure is caught early, at `data_load_images`, with a clear,
  specific cause (wrong image dimensions) rather than later as a
  generic `input_size` mismatch at the `infer_run` boundary with less
  context about why.
- Future loaders for any other fixed-shape file format in this project
  should default to the same posture: validate against this project's
  actual, current architectural assumptions, not only against the
  general file format's own minimal validity rules — the format being
  technically well-formed is necessary but not sufficient for this
  project's purposes.
