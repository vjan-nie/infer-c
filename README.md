# infer-c

A from-scratch neural network inference engine in pure C99. Given a
pre-trained two-layer MLP and a handwritten digit image, it classifies
the digit with 92.76% accuracy on the MNIST test set — no ML framework,
no external dependencies, and no Python required at inference time.

---

## Quick start

```bash
git clone <this-repo>
cd infer-c
make
bin/infer weights.bin <your-image.idx>
```

`weights.bin` is committed to the repository (see [ADR-014](docs/adr/ADR-014-versioned-pretrained-weights.md)).
A pre-trained model is available immediately after cloning — no Python or PyTorch
installation needed to run `bin/infer`. To retrain from scratch, see
[Training](#training).

`<your-image.idx>` must be a single-image IDX3 file (28×28 pixels, the
standard MNIST format). `bin/infer` prints an ASCII rendering of the digit
followed by the predicted class and confidence:

```
            ####
            ####
            ####
            ####
            ####
            ####
            ####
            ####
            ####
            ####
            ####

Predicted digit: 1
Confidence: 92.2%
```

---

## Architecture

```
python/train.py          trains the MLP, exports weights.bin
      │
      │  (binary file contract — IC01 format, ADR-004)
      ▼
src/model.c              loads weights.bin; validates CRC32, magic, format version
src/nn.c                 forward pass: mat_mul → relu → mat_mul → softmax
src/matrix.c             row-major Matrix type and ops (mat_mul, mat_add_bias, ...)
      │
      │  (public API boundary — ADR-001, ADR-012)
      ▼
src/infer_c.h            opaque InferModel*, infer_model_load, infer_run
      │
      ├──► src/main.c       single-image CLI demo (bin/infer)
      └──► src/benchmark.c  batch accuracy tool (bin/benchmark)
```

**Python/C split:** Python (PyTorch) handles training. C handles inference
only. The boundary is `weights.bin`: `train.py` writes it once; the C engine
reads it on every run. The C side has no training loop, no autodiff, and no
Python dependency at runtime.

---

## Verified results

| Check | Result |
|---|---|
| MNIST test-set accuracy | 9276 / 10000 (92.76%) |
| Per-sample logit agreement with Python reference | max diff < 5e-6 (float32 rounding) |
| Per-sample softmax agreement | max diff < 1e-7 |
| Valgrind (test suite, 4 binaries) | 0 leaks, 0 errors |
| Valgrind (`bin/infer` — single image) | 0 leaks, 0 errors |
| Valgrind (`bin/benchmark` — 10,000 images) | 0 leaks, 0 errors |

The logit and softmax checks run as part of `make test` (`test_nn` and
`test_infer_c`) against 20 verification samples exported from the same
`train.py` run that produced `weights.bin`.

---

## Build, test, benchmark

```bash
make            # build everything: test suite, lib/libinfer_c.{a,so}, bin/infer, bin/benchmark
make test       # run the assert-based test suite (4 modules)
make valgrind   # run the test suite under valgrind --leak-check=full
make benchmark  # run bin/benchmark against data/raw/ (requires MNIST IDX files)
make clean      # remove bin/ and lib/
```

`make` requires only a C99 compiler (`cc`). Zero external libraries — `libm`
is linked for `expf` in the softmax layer; everything else is standard C99.

---

## Training

To reproduce `weights.bin` from scratch:

```bash
# Download MNIST
bash python/download_mnist.sh            # saves to data/raw/

# Install Python dependencies
cd python && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train (seed 42, ~30 seconds on CPU)
python train.py
```

`train.py` exports `weights.bin` and `verification_samples.bin` to the repo
root. The training is deterministic (seed 42, fixed architecture) — the
resulting `weights.bin` is byte-for-byte identical to the committed one.

---

## Design decisions

Fourteen Architecture Decision Records live in [`docs/adr/`](docs/adr/). A
few worth highlighting:

**Weight transpose at load time** (`model.c`, discovered in M2). PyTorch
stores weight matrices as `out_features × in_features`. The C forward pass
computes `output = input_row × W`, which requires `W` to be
`in_features × out_features`. `model_load` transposes each weight matrix once
at load time — the forward pass never has to think about orientation. The
transpose is a `static` helper inside `model.c`; there is no second use case
for a generic transpose, so it was not promoted to a `matrix.h` primitive.

**Public API error boundary** ([ADR-012](docs/adr/ADR-012-public-api-error-boundary.md)).
Internal code uses `assert` for programmer errors and `NULL` for environment
errors. At the public API (`infer_c.h`) a third rule applies: no caller
mistake ever triggers `assert` or any other process-aborting behavior. A wrong
`input_size` passed to `infer_run` returns a sentinel
`{ .predicted_class = -1 }` — a mistake by external code linking this library
can never take down the host process.

**CRC32 before parsing** ([ADR-010](docs/adr/ADR-010-byte-order-checksum-scope.md)).
`model_load` verifies the file's CRC32 over its entire byte range *before*
inspecting `magic`, `format_version`, or any structured field. Checking
structured fields before the checksum risks following a corrupted
`format_version` down the wrong parsing path before the integrity check runs.

---

## Scope

This is an inference engine for one fixed architecture (784→128→10 MLP,
ReLU hidden layer, softmax output) trained on MNIST. It is not a general
neural network framework — there is no graph executor, no operator registry,
and no support for architectures other than the one it was built for.

Explicitly out of scope (not unfinished, deliberately deferred):

- Convolutional layers
- SIMD / hardware acceleration
- int8 / quantized inference
- ONNX or `.npy` format support
- Cross-endian portability
- GPU execution
