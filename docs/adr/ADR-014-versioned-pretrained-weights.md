# ADR-014 — Versioning a Pre-Trained `weights.bin` for Immediate Usability

## Status
Accepted

## Context

M1 added `weights.bin`/`verification_samples.bin` to `.gitignore`, with
the stated rationale: "Generated model artifacts — regenerate with
python/train.py... not meant to be versioned as source," since training
is deterministic (fixed seed, ADR-009) and anyone can reproduce the
exact same file by running `train.py`.

That reasoning is still correct in principle, but M4 — the project's
final polish milestone, oriented toward portfolio presentation — surfaces
a practical cost it didn't account for: a person evaluating this repo
(a recruiter, an interviewer, a curious engineer) who wants to actually
run `bin/infer` or `bin/benchmark` and see real output must first set up
a Python environment, install PyTorch, download the MNIST IDX files, and
run a multi-minute training script — entirely before touching any of the
C code this project is actually about. That's a significant amount of
friction between "I want to see if this works" and seeing it work, for a
project whose primary deliverable is the C inference engine, not the
training pipeline.

## Decision

A pre-trained `weights.bin` (and its paired `verification_samples.bin`)
is committed to the repository as a versioned artifact, reversing M1's
`.gitignore` exclusion for these two specific files.

This is narrowly scoped: it does not version `data/` (the MNIST dataset
itself — large, third-party, redistribution is a separate question this
ADR doesn't reopen) or any other generated artifact. Only the two small
files (~470KB combined) needed to run the C engine immediately after
cloning the repo.

Rationale for the reversal: M1's reasoning ("anyone can regenerate it")
remains true and is still the right default posture for most generated
artifacts — but it optimizes for a contributor who is going to modify
the training pipeline, not for an evaluator who wants the fastest path
to "does this actually work?" For a portfolio project, the second
audience matters more than M1 anticipated. Determinism (the same
`train.py` run reproduces this exact file) is what makes versioning it
safe: it is not a divergent, hand-edited, or environment-specific
artifact that could drift from what `train.py` would produce — it's a
convenience copy of something fully reproducible, not a second source of
truth.

## Consequences

- `.gitignore`'s exclusion of `weights.bin`/`verification_samples.bin`
  is removed for these two filenames specifically. Any *other* generated
  artifact (build outputs in `bin/`/`lib/`, future re-training outputs
  under a different name) remains gitignored as before — this ADR does
  not change the project's general posture on build artifacts.
- The README (M4) should state clearly that the committed `weights.bin`
  is provided for convenience and was produced by the exact `train.py`
  run documented in M1 (seed 42, 10 epochs, 92.76% test accuracy) — not
  imply it's hand-tuned or differs from what running `train.py` locally
  would produce.
- If `train.py`, the model architecture, or the weight format ever
  change in a way that makes the committed file stale or incompatible
  (a `format_version` bump, ADR-004), the committed `weights.bin` must
  be regenerated and re-committed in the same change — it is not
  acceptable for the versioned artifact to silently drift out of sync
  with the code that produces and consumes it.
- This does not relax M1's underlying principle for any other artifact
  in this project; it is a one-time, narrowly-justified exception for
  exactly two files, made explicit here so it doesn't read as an
  inconsistency when read against ADR-history.
