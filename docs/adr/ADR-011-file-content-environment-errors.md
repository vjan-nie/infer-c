# ADR-011 — File-Content-Derived Values Are Always Environment Errors

## Status
Accepted

## Context

M2's adversarial review found that `model_load`'s chain-compatibility
check used `assert` on the reasoning that "by this point in parsing, an
inconsistency can only be a programmer error" (the original audit's
claim, M2's `01-audit.md`). That reasoning was wrong: CRC32 is not
cryptographic, so a file can have a valid checksum and still contain
internally inconsistent values (e.g. mismatched layer dimensions) —
the checksum proves the bytes weren't accidentally corrupted in transit,
it says nothing about whether those bytes encode a self-consistent model.

The same review traced a second, related gap: a per-layer size-bound
check that could itself be bypassed by integer overflow on
attacker/corruption-controlled `rows`/`cols` values, again routing what
should be an environment error into an `assert`-abort path via a
downstream cast.

Both were fixed in M2b by switching the affected checks from `assert` to
`free` + `return NULL`, matching `model_load`'s existing documented
contract.

## Decision

**Any value that originates from parsed file content remains an
environment-error category for the entire duration it is used —
regardless of how many validation steps already passed, and regardless
of how far downstream in the parsing logic it is consumed.**

ADR-005's original assert-vs-NULL distinction (programmer error vs.
environment error) is correct and unchanged by this ADR. What this ADR
clarifies is a boundary case that distinction didn't explicitly cover:
"a value passed an earlier check" does not promote it from
environment-derived to programmer-controlled. Only values that are
*never* read from external input — true internal invariants the code's
own logic guarantees — qualify for `assert`.

Concretely, for `model_load` and any future parser in this project
(`data.c` in M3 will read user/file-provided IDX content under the same
discipline):

- A checksum, magic number, or format-version check passing does **not**
  certify that every subsequently-derived value is now safe to trust
  unconditionally. It only certifies that those specific bytes are
  present and well-formed at that check's specific scope.
- Any arithmetic performed on a file-derived value (size calculations,
  offset bookkeeping, index bounds) must be checked for overflow/bounds
  *before* being trusted, not assumed safe because "the file passed
  validation already."
- The test for "is this an environment error?" is not "where in the
  function does this check occur?" but "could a legitimately-existing,
  externally-supplied file (valid or corrupted, accidentally or
  deliberately) cause this condition?" If yes, it's an environment
  error → `NULL`, no matter how late in parsing it's discovered.

## Consequences

- `model_load` now has zero `assert`-abort paths reachable by file
  content alone (M2b's fix). Any future module that parses external
  binary input (`data.c`, M3) should be designed against this same
  discipline from the start, not audited for it after the fact.
- This does not change the cost/benefit calculus around CRC32 itself
  (ADR-008/ADR-010) — the checksum remains valuable for catching
  accidental corruption, which is its stated purpose. This ADR only
  corrects the inference that a passing checksum implies anything
  beyond "these specific bytes are intact."
- Future reviews (adversarial or self-review) should specifically ask,
  for every `assert` in file-parsing code: "could a file constructed by
  someone who knows the format, but not the code's internal invariants,
  reach this condition?" If the answer isn't a confident no, it's
  miscategorized.
