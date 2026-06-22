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
│   ├── nn.h / nn.c           # placeholder: NN layers, forward pass (M2)
│   ├── model.h / model.c     # placeholder: weight file loading (M1/M2)
│   ├── data.h / data.c       # placeholder: MNIST data loading (M1/M2)
│   ├── infer_c.h             # placeholder: public API entry point (M2.5)
│   └── main.c                # placeholder: CLI demo entry point (M2)
├── python/
│   └── train.py              # placeholder: trains MLP, exports weights (M1)
├── tests/
│   └── test_matrix.c         # assert-based test runner for matrix.c
├── docs/
│   └── adr/                  # Architecture Decision Records
├── .claude/                  # workflow + skills
├── Makefile
└── CLAUDE.md                 # this file
```

`infer_c.h` is the planned future public-API header (see ADR-001): it
deliberately stays empty until `nn.h`/`model.h` are stable, because its
shape depends on what those modules end up exposing.

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
- Build: `make` (currently an alias for `make test` — see §6.1)
- Tests: `make test`
- Memory: `make valgrind`
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

- The `all` target is currently an alias for `test` (see ADR-006). `nn.c`,
  `model.c`, `data.c`, and `main.c` are intentionally empty placeholders
  (M0 scope); an empty `.c` translation unit triggers
  `warning: ISO C forbids an empty translation unit [-Wpedantic]` if
  compiled. Do not add them to the Makefile's build list until they have
  real content — a real `all` target (linking a demo binary) returns in
  M2 once `main.c` has an actual `main()`. Do not "fix" the warning by
  adding dummy content to a placeholder or relaxing `-Wpedantic` — ADR-006
  considered and rejected both.

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

### 6.3 Testing allocation-failure paths

- To deliberately trigger a `malloc` failure in a test (e.g. for
  verifying a `NULL`-on-failure contract), keep the requested byte count
  under 2⁶³. Valgrind's memcheck flags any `malloc` size at or above
  2⁶³ as "fishy (possibly negative)" and fails the run, even when there
  is no actual undefined behavior or leak — this triggered a false
  positive when `mat_create(INT_MAX, INT_MAX)` was first tried (M0b).
  A request like `mat_create(1000000000, 1000000000)` (~4 exabytes) is
  still unsatisfiable on any real system, exercises the same code path,
  and stays under Valgrind's threshold.

---

## 7. When you are unsure

Ask. A clarifying question costs one round-trip; a wrong assumption costs
an hour of rework.
