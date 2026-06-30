# CLAUDE.md — infer-c

This file is read by Claude Code at the start of every session. Keep it
short. When Claude makes a mistake and you correct it, add a rule here so
it does not repeat.

This project follows the three-agent workflow defined in `WORKFLOW.md`.

---

## 0. Language rule (non-negotiable)

All code, comments, commit messages, identifiers, and documentation in
this repository must be written in **English**. This applies to every
file, present and future, without exception.

---

## 1. Engineering mindset (non-negotiable)

Apply the `critical-software-engineering` skill at
`.claude/skills/critical-software-engineering/SKILL.md`.

Summary:
- Think before coding. State assumptions.
- Simplicity first. No abstractions for single-use code.
- Surgical changes. Touch only what you must.
- Goal-driven execution. Tests before/after.
- Honesty over helpfulness.

---

## 2. Repository map

```
infer-c/
├── src/
│   ├── matrix.h / matrix.c   # implemented: row-major Matrix + ops (M0)
│   ├── model.h / model.c     # implemented: weight file loading (M2)
│   ├── nn.h / nn.c           # implemented: NN layers, forward pass (M2)
│   ├── data.h / data.c       # implemented: IDX image/label loading (M3)
│   ├── infer_c.h / infer_c.c # implemented: public API, opaque InferModel (M2.5)
│   ├── main.c                # implemented: single-image CLI demo (M3)
│   └── benchmark.c           # implemented: batch accuracy tool (M3)
├── python/
│   └── train.py              # trains MLP, exports weights.bin/verification_samples.bin (M1)
├── tests/
│   ├── test_matrix.c         # assert-based test runner for matrix.c
│   ├── test_nn.c             # assert-based test runner for model.c + nn.c
│   ├── test_infer_c.c        # assert-based test runner for infer_c.c (public API only)
│   └── test_data.c           # assert-based test runner for data.c
├── docs/
│   └── adr/                  # Architecture Decision Records
├── .claude/                  # workflow + skills
├── Makefile
└── CLAUDE.md                 # this file
```

`infer_c.h` is the project's public-API header (ADR-001): an external
consumer only needs this header and `lib/libinfer_c.a`/`libinfer_c.so`
— `matrix.h`/`model.h`/`nn.h` stay internal to this repo's own build.

---

## 3. Stack and conventions

- **Language:** C99 (`-std=c99`).
- **Compiler:** `cc`, flags `-Wall -Wextra -Wpedantic`. Zero warnings is a
  hard gate, not aspirational — there's no legacy debt to excuse it.
- **Numeric type:** `float` (32-bit) throughout (ADR-005). Never `double`.
- **Naming convention:** short prefix `mat_` for matrix functions
  (`mat_create`, `mat_mul`, ...), not `matrix_` (ADR-005). Future modules
  (`nn_`, `model_`, `data_`) should follow the same short-prefix pattern.
- **Matrix representation:** row-major flat array (`float* data`, `rows`,
  `cols`), one `malloc` per matrix (ADR-002). Never an array of pointers.
- **Error handling:** dual convention (ADR-005) — `assert()` for
  programmer errors (incompatible dimensions), `NULL` data pointer for
  environment errors (`malloc` failure). Each dimension check used by an
  `assert` is factored into its own pure `mat_dims_compatible_for_*`
  predicate so it's directly testable without aborting the test binary.
  At the public API boundary (`infer_c.h`), a third category applies
  instead (ADR-012): never `assert`/abort on caller-supplied input —
  return a sentinel `InferResult` (`predicted_class == -1`) so a
  mistake by external code linking this library can never take down
  the host process. This is the default for any future public API
  addition, not just `infer_run`.
- **Build flags:** every object file is compiled with `-fPIC`
  unconditionally — one object set serves both `lib/libinfer_c.a` and
  `lib/libinfer_c.so`, no separate PIC/non-PIC build.
- **Testing:** no external framework. A plain `assert`-based runner per
  module (`tests/test_<module>.c`), one `test_<case>(void)` function per
  case, called sequentially from `main()`.
- **Memory safety:** every test run is expected to be clean under
  `valgrind --leak-check=full`. Zero leaks is a hard gate.

---

## 4. How to work in this repo

### 4.1 Before touching code
1. Read the file you will change. Do not edit blind.
2. Check `docs/adr/` for a binding decision before introducing a new
   convention (numeric type, naming, error handling, file formats).
3. Check `.claude/workflow/tasks/` for context on recent decisions.

### 4.2 Verifying changes
- Build: `make` (runs tests, then builds `lib/libinfer_c.a`/`.so` and `bin/infer`)
- Tests: `make test`
- Memory: `make valgrind`
- Library only: `make lib`
- Accuracy benchmark: `make benchmark` (requires `data/raw/` IDX files; not in `make test`)
- Clean: `make clean`

### 4.3 Pull request hygiene
- One concern per PR. Unrelated bugs go in a sibling task.
- Commit messages: imperative, scoped, in English.

---

## 5. Three-agent workflow

Workflow documents live at:
```
.claude/workflow/tasks/<TASK_SLUG>/
├── 01-audit.md
├── 02-plan.md
├── 03-diff.patch
└── 04-review.md
```

Create the folder with `./.claude/workflow/init-task.sh <TASK_SLUG>`.

---

## 6. Known traps in this codebase

### 6.1 Build

- All source files under `src/` now have real content and are compiled.
  If a future placeholder is added, remember: an empty `.c` translation
  unit triggers `warning: ISO C forbids an empty translation unit
  [-Wpedantic]` if compiled — do not add it to the Makefile until it has
  real content, and do not "fix" the warning with dummy content or by
  relaxing `-Wpedantic` (ADR-006).

### 6.2 Matrix module

- `mat_mul`'s `out` must be pre-created by the caller with the correct
  dimensions (`a->rows x b->cols`); `mat_mul` never allocates. Same for
  `mat_copy`'s `dst`.
- `mat_add_bias` broadcasts a `1 x cols` bias row across every row of `m`
  — bias is never a column vector or a per-matrix scalar.
- The `assert`-abort paths inside `mat_mul`/`mat_add_bias` are not
  exercised by automated tests (would abort the whole test binary).
  Coverage of that validation logic comes from testing the
  `mat_dims_compatible_for_*` predicates directly, with valid and invalid
  inputs.

### 6.3 Model loading and forward pass

- `weights.bin` stores each layer's weight as `rows x cols` =
  `out_features x in_features` (ADR-004). `model_load` transposes each
  weight matrix at load time into `in_features x out_features` — the
  orientation the forward pass needs for `mat_mul(input_row, W)` to
  type-check and for `mat_add_bias` to broadcast correctly. The
  transpose helper is `static` to `model.c`, not a `matrix.h` primitive
  — there is no second use case for a generic transpose yet.
- `model_load` verifies the file's CRC32 over its entire byte range
  (minus the trailing 4-byte checksum field) *before* interpreting
  `magic`/`format_version`/anything else (ADR-010). Do not reorder this:
  checking structured fields before the checksum risks following a
  corrupted `format_version` down the wrong parsing path before the
  checksum check ever runs.
- `model_crc32` is table-driven (256-entry `static const` lookup table,
  standard CRC-32/ISO-HDLC, poly `0xEDB88320` reflected) — not
  bit-by-bit. An earlier audit draft proposed bit-by-bit "for
  auditability"; that was reconsidered in M2's plan phase: a lookup
  table is the de facto standard (zlib itself uses one) and is no less
  auditable when its construction is documented in a comment.
- `model_load` returns a heap-allocated `Model*`; `model_free` frees the
  layers, their matrices, and the `Model` struct itself in one call —
  there is no separate `free()` step for the caller.

### 6.4 Testing allocation-failure paths

- To deliberately trigger a `malloc` failure in a test (e.g. for
  verifying a `NULL`-on-failure contract), keep the requested byte count
  under 2⁶³. Valgrind's memcheck flags any `malloc` size at or above
  2⁶³ as "fishy (possibly negative)" and fails the run, even when there
  is no actual undefined behavior or leak — this triggered a false
  positive when `mat_create(INT_MAX, INT_MAX)` was first tried (M0b).
  A request like `mat_create(1000000000, 1000000000)` (~4 exabytes) is
  still unsatisfiable on any real system, exercises the same code path,
  and stays under Valgrind's threshold.

### 6.5 Workflow document gaps (known, explained)

- `04-review.md` is empty for tasks that never had an independent
  Phase 3 (M0, M0b, M2b) — these were either reviewed critically
  in-chat instead of via a separate agent session, or were small
  sibling-task corrections scoped to skip a full three-phase cycle.
- `01-audit.md` is empty for M0b/M2b specifically — both combined
  Audit+Plan+Execute into a single phase (deliberate, proportional to
  a small scoped fix), so the audit content lives inside `02-plan.md`'s
  Context section instead of its own file.
- `02-plan.md`/`03-diff.patch` are empty for M2-5-public-api
  specifically — this one is a genuine process gap (the plan/diff were
  approved and executed but never written back to these files), not a
  deliberate skip. Left as a known gap; the actual plan content is
  preserved in the conversation history with the project's orchestrator
  rather than in-repo.

### 6.6 Big-endian vs. little-endian convention split between `data.c` and `model.c`

- `data.c` reads IDX headers with an explicit big-endian helper
  (`read_be_u32`, bit-shift MSB-first). `model.c` reads IC01 headers
  with `memcpy`-based little-endian reads (the file format and host are
  both little-endian by assumption, ADR-010). These two conventions live
  in adjacent files and must not blur: a future edit touching both
  modules must not "fix" one convention by applying the other.

### 6.7 `data_load_images` validates architectural assumptions, not just format validity (ADR-013)

- `data_load_images` rejects any IDX image file whose `rows`/`cols` are
  not exactly `28×28`, even though the IDX format itself places no such
  constraint. This is intentional: the trained model's first layer
  (`784 → 128`) cannot accept any other input width. A structurally
  valid IDX file with different dimensions is not valid input for this
  project's architecture. Future loaders for fixed-shape data should
  default to the same posture — validate against the current
  architectural assumption, not only the file format's minimal validity
  rules.

### 6.8 Makefile: phony target name colliding with an existing directory

- A target named exactly like an existing directory (e.g. `lib` as both
  a Makefile target and the `lib/` output directory) can be silently
  treated by Make as "already up to date" — Make sees the directory's
  mtime and skips the recipe entirely, even though no library was ever
  built. Always list any such target in `.PHONY` (e.g.
  `.PHONY: all test valgrind clean lib`) so Make never treats the
  directory's existence as satisfying the target. Found and fixed
  during M2.5 (`lib/libinfer_c.a`/`.so` build); confirmed present in the
  current Makefile's `.PHONY` line during M2.5's adversarial review.

---

## 7. When you are unsure

Ask. A clarifying question costs one round-trip; a wrong assumption costs
an hour of rework.
