# ADR-006 — Deferred `all` Target vs. Empty Placeholder Files

## Status
Accepted

## Context

M0's scope intentionally creates placeholder files for modules not yet
implemented (`nn.c`, `model.c`, `data.c`, `main.c`) — empty except for a
one-line comment stating their future purpose. This is a deliberate
decision (see M0 plan, §1) to make the directory structure complete from
the first commit without implementing logic ahead of its own milestone.

During Phase 2 planning, compiling an empty translation unit under this
project's mandatory flags (`-std=c99 -Wall -Wextra -Wpedantic`) was found to
produce:

```
warning: ISO C forbids an empty translation unit [-Wpedantic]
```

This puts two of the project's own commitments in direct conflict:

- Zero warnings, as a hard gate, from day one (stated in the M0 audit and
  treated as non-negotiable since there is no legacy debt to excuse it).
- A complete directory scaffold with placeholder files for not-yet-built
  modules.

The Makefile's `all` target, as sketched in the audit, links
`matrix.o nn.o model.o data.o main.o` into a demo binary — which would
compile the empty placeholders and trip the warning.

## Options considered

**A — Add dummy content to each placeholder** (e.g. an unused `typedef` or
a `void unused(void) {}`) purely to silence the empty-translation-unit
warning.

Rejected: this is exactly the kind of "comment that explains what, not why"
anti-pattern the project's engineering posture warns against — dummy code
added only to satisfy a tool, with no relationship to the module's actual
future purpose. It would also need to be deleted, not extended, once real
implementation starts in M1/M2, which is its own small source of churn and
confusion for no benefit.

**B — Relax the warning gate** (e.g. drop `-Wpedantic`, or suppress this
specific warning).

Rejected outright: the zero-warnings gate is a stated project commitment,
not a default to be weakened the first time it's inconvenient. Weakening a
quality gate to accommodate a placeholder is the wrong direction of
pressure — the placeholder strategy should adapt, not the gate.

**C — Defer the `all` target.** Until placeholder modules have real
implementations (M2), `all` is an alias for `test`: only `matrix.c` (the
only module with real code this milestone) is compiled and linked. The
real, multi-module `all` target returns once `nn.c`/`model.c`/`data.c`/
`main.c` stop being empty placeholders.

Chosen.

## Decision

`make all` is an alias for `make test` until M2. The Makefile does not
attempt to compile or link the placeholder `.c` files in any target during
M0 and M1. This is documented in the Makefile itself (a comment at the
`all` target) and in the M0 plan, so it reads as an intentional, temporary
state — not an oversight or a silently abandoned target.

## Consequences

- Anyone running `make` during M0/M1 builds and runs the test suite, not a
  (nonexistent yet) demo binary. This is correct: there is nothing to
  demo yet.
- When M2 gives `nn.c`/`model.c`/`data.c`/`main.c` real implementations,
  the `all` target must be revisited and restored to actually build the
  demo binary — this is a known, tracked follow-up, not a new discovery.
- This sets a precedent for how this project resolves tooling/scope
  conflicts in general: the quality gate stays fixed, the build strategy
  adapts around it. Future similar conflicts should default to the same
  resolution direction.
