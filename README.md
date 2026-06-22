# infer-c

A small neural-network inference engine written from scratch in pure C99,
with no external dependencies. The goal is to load a pre-trained MLP and
run its forward pass natively in C.

## Current state

This is an early-stage scaffold (milestone M0). Only the matrix module
(`src/matrix.h` / `src/matrix.c`) is implemented: a row-major `Matrix`
type with `mat_create`, `mat_free`, `mat_mul`, `mat_add_bias`, `mat_copy`,
and the dimension-compatibility predicates used internally and by tests.

There is no model loading, no forward pass, and no training script yet —
`nn.c`, `model.c`, `data.c`, `main.c`, and `python/train.py` exist only as
placeholder files reserving their place in the directory structure.

## Build

```bash
make        # builds and runs the test suite (no demo binary yet, see below)
make test   # same as above, explicit
make clean  # removes build artifacts
```

`make` currently builds and runs the test suite rather than a demo binary,
because the modules a runnable demo depends on (`main.c`, `nn.c`,
`model.c`, `data.c`) are not implemented yet.

## Test

```bash
make test       # runs the assert-based test suite for matrix.c
make valgrind   # runs the same suite under valgrind --leak-check=full
```

## Design decisions

See `docs/adr/` for the Architecture Decision Records behind this
project's structure (matrix representation, weight file format, numeric
type, naming convention, error handling).
