# ADR-002 — Matrix Representation: Row-Major Flat Array

## Status
Accepted

## Context

The core data structure of `infer-c` is a matrix/vector type used to
represent weights, biases, and activations throughout the forward pass. Two
common representations exist in C:

- A flat, contiguous array (`float* data`) addressed as `data[row * cols +
  col]`, with `rows` and `cols` stored alongside it.
- An array of pointers (`float** data`), where each `data[row]` points to an
  independently allocated row.

This choice affects performance, memory-safety surface, and the complexity
of every operation built on top of it (`mat_mul`, `mat_add_bias`, etc.), so
it needs to be fixed before any of those operations are implemented.

## Options considered

**A — Row-major flat array.**

```c
typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;
```

A single `malloc` of `rows * cols` floats; element access via
`data[row * cols + col]`.

**B — Array of pointers.**

```c
typedef struct {
    float** data;
    int rows;
    int cols;
} Matrix;
```

One `malloc` for the array of row pointers, plus `rows` further independent
`malloc` calls, one per row.

## Decision

Option A — row-major flat array.

Rationale:

- **Cache behavior.** The forward pass's dominant access pattern is
  iterating over a full row of weights per output neuron. A flat array
  makes this a sequential memory scan; an array of pointers adds a pointer
  dereference per row and scatters rows across the heap, with no guarantee
  of locality between them.
- **Memory-safety surface.** A flat array requires exactly one allocation
  and one matching free per matrix. An array of pointers requires `rows +
  1` allocations and frees, all of which must succeed and be freed
  correctly for the structure to be leak-free — `rows` times more
  opportunities for a partial-allocation failure or a missed free,
  multiplied across every matrix the program creates.
- **Simplicity.** This project's `critical-software-engineering` posture
  favors the simpler structure when it isn't outperformed on its own
  merits by the more complex one — here the flat array is both simpler
  *and* faster, so there's no trade-off to weigh.
- **Industry precedent.** This is the representation NumPy, BLAS, and most
  lightweight inference engines use internally, for the same reasons.

## Consequences

- Every matrix operation (`mat_create`, `mat_mul`, `mat_add_bias`,
  `mat_copy`, etc.) indexes `data` as `data[row * cols + col]`. This
  indexing arithmetic is the one place where an off-by-one or
  row/column-order mistake is most likely; it should be isolated in as few
  places as possible (ideally a single indexing helper or macro used
  consistently, rather than re-derived ad hoc in every function).
- `mat_free` only ever needs to free a single pointer (`data`) plus the
  struct itself, if heap-allocated — there is no nested loop of frees to
  get wrong.
- This representation is assumed by ADR-004 (weight file format): weights
  are stored and read back in row-major order, with no transposition step
  needed between the file and in-memory layout, *provided* `train.py`
  exports them in the same order (flagged as an open risk in the M0 audit,
  to be confirmed in M1).
