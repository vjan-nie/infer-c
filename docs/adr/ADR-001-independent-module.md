# ADR-001 — infer-c as an Independent, Self-Contained Module

## Status
Accepted

## Context

This project originated as a portfolio piece: a small inference engine in
pure C99, with no ML frameworks, that loads a pre-trained MLP and classifies
MNIST digits. Separately, there is a possibility that this engine could
later be linked into another, unrelated system as its low-level inference
component.

That other system is proprietary, cannot be shown publicly, and its
existence must not leak into this repository in any form — no naming, no
comments, no design choices that only make sense in light of it, no
references in commit messages or documentation.

At the same time, the engine's public API should be designed so that, if
that future integration happens, it requires no rework of infer-c itself —
only a thin adapter on the other side of the boundary.

## Decision

`infer-c` is designed, documented, and presented as a fully independent,
generic inference engine. Specifically:

- No file, identifier, comment, commit message, or document in this
  repository refers to any external system, domain, or integration target.
- The public API (to be defined in `infer_c.h`, M2.5) operates exclusively
  on generic types: a flat `float*` array as input, an opaque model handle,
  a plain `{predicted_class, confidence}` result. It does not encode any
  concept (data framing, protocol, message types) belonging to a specific
  external system.
- Any translation between an external system's own data representation and
  the flat array this library expects is explicitly the responsibility of
  whoever links against this library — never something infer-c implements
  or assumes.
- The project's narrative (README, this ADR series) presents infer-c on its
  own merits: systems-level C implementation, clean memory management,
  numerical correctness verified against a Python reference, zero
  dependencies. It does not require, reference, or depend on any external
  context to be understood or evaluated.

## Consequences

- Every later design decision (weight format, public API shape, build
  artifacts) is evaluated first against "does this make sense for a
  generic, standalone inference engine?" — not against the needs of any
  hypothetical integration.
- This rules out, by design, certain conveniences an integration-aware
  design might otherwise take (e.g. a model format tied to a specific
  external schema, or an API shaped around a specific consumer's data
  layout). This is an accepted cost in exchange for the module being
  genuinely presentable as independent portfolio work.
- The public API surface (M2.5) must be reviewed against this ADR
  specifically: if a proposed function signature only makes sense assuming
  a particular external consumer, that's a violation of this decision and
  should be flagged before merging.
- This ADR does not forbid the library from being used by any external
  system later — it only forbids that future use from shaping or appearing
  in this codebase.
