# From Zero to a Working Neural Network: A Learning Guide Built Around infer-c

This guide explains, from first principles, every concept needed to
understand `infer-c` — a small neural network inference engine written in
pure C. It assumes no prior machine learning background. If you already
know some of this, skip ahead; each section is written to stand mostly on
its own.

A note on scope: this project is **computer vision** (classifying images
of handwritten digits), not natural language processing. The two fields
share the same underlying building block — the feedforward neural
network — but NLP adds its own vocabulary on top (tokens, embeddings of
words, attention, transformers) that this project does not use and this
guide does not cover. Everything here is genuinely foundational, though:
the MLP (multi-layer perceptron) you'll understand by the end of this
guide is the same kind of building block that shows up, in more elaborate
forms, inside almost every modern ML system, NLP included.

---

## Part 1 — What problem are we even solving?

### The task: MNIST digit classification

MNIST is a dataset of 70,000 small grayscale images, each one a single
handwritten digit (0 through 9), each image exactly 28×28 pixels. The
task is: given one of these images, output which digit it is.

This is a **classification** problem: the output is one of a fixed,
small set of categories (here, 10 — the digits 0–9), not a continuous
number. Contrast this with **regression**, where the output would be a
continuous value (e.g. predicting a house price). Classification and
regression are the two broad families most supervised learning problems
fall into.

### Why this is a good first project

MNIST is small, well-understood, and the input is simple (just pixel
brightness values — no color, no complex structure). It's the
"hello world" of machine learning for a reason: it's complex enough to
need a real neural network, simple enough that you can fully understand
every number flowing through the system.

---

## Part 2 — Representing an image as numbers

### Pixels are just numbers

A 28×28 grayscale image is, underneath, a grid of 784 numbers (28×28 =
784), each one a brightness value. In the raw MNIST file format, each
pixel is stored as a single byte (`uint8_t` in C), ranging from 0 (black)
to 255 (white).

### Flattening

Neural networks of the simple kind used here don't think in terms of
"rows and columns" — they take a flat list of numbers as input. So the
28×28 grid gets **flattened** into a single sequence of 784 numbers,
read row by row (row 0's 28 pixels, then row 1's 28 pixels, and so on).
This is exactly what `infer-c`'s `data_load_images` function does: it
reads a 28×28 grid from a file and hands back a flat array of 784
floating-point numbers.

The flattening throws away some structure — the network no longer
"knows," by the shape of the data alone, which pixels were
spatially adjacent in the original image. This is one of the real
limitations of the architecture used here (a plain MLP) compared to a
**convolutional neural network** (CNN), which is specifically designed
to preserve and exploit 2D spatial structure. infer-c deliberately does
not implement convolutions — this is listed as an explicit, honest
scope boundary in the project's README, not a missing feature nobody
thought of. A plain MLP can still do well on MNIST (this project reaches
92.76% accuracy) precisely because MNIST is simple enough that spatial
structure matters less than it would for, say, photographs of real-world
objects.

### Normalization

Raw pixel values (0–255) get divided by 255 before being fed into the
network, producing values in the range [0.0, 1.0] instead. This is
called **normalization**.

Why bother? Neural networks are built from repeated multiplication and
addition (you'll see exactly how shortly). If input values are large
(like 255), the numbers flowing through the network during training tend
to grow very large or very small in ways that make learning slower and
less stable. Keeping inputs in a small, consistent range (like [0, 1])
makes the whole numerical process much better-behaved. This is a
near-universal practice across machine learning, not specific to images
or to this project.

---

## Part 3 — What is a neural network, actually?

### The biological inspiration (and why it's just inspiration)

The name "neural network" comes from a loose analogy to biological
neurons: a neuron receives signals from other neurons, combines them, and
if the combined signal is strong enough, it "fires" and sends a signal
onward. Artificial neural networks borrow this rough shape but are, in
practice, just a particular kind of mathematical function — a chain of
matrix multiplications and simple nonlinear operations. You don't need
the biology to understand the math; the biological framing is mostly
useful as a memory aid and a piece of the field's history.

### A single neuron: weighted sum + bias

The smallest building block is a single **neuron** (also called a
**unit**). It takes some number of inputs, multiplies each by its own
**weight**, adds them all together, and adds one more number called the
**bias**:

```
output = (input_1 × weight_1) + (input_2 × weight_2) + ... + (input_n × weight_n) + bias
```

That's it — a weighted sum, plus a constant offset. The weights and bias
are the neuron's **parameters**: the numbers that get adjusted during
training so the network's overall output becomes useful. Before training,
weights and biases typically start out as small random numbers, which is
why an untrained network's output is just noise. After training, they
encode whatever pattern the network learned.

### A layer: many neurons side by side

A **layer** is a group of neurons that all see the same input and each
compute their own weighted sum + bias, independently, in parallel. If a
layer has, say, 128 neurons, and the input has 784 numbers, then that
layer has 784 × 128 weights (one weight per input-neuron pair) plus 128
biases (one per neuron) — 100,480 numbers total, all of them learned
during training.

This is exactly why `infer-c` represents weights as a **matrix**: a grid
of numbers where each row (or column, depending on convention — more on
this below) holds one neuron's set of weights. Computing an entire
layer's output for a given input turns out to be exactly **matrix
multiplication** — which is why `infer-c`'s core data structure,
`Matrix`, and its central operation, `mat_mul`, exist. The neural network
math and the linear algebra are the same thing, just described in two
different vocabularies.

### A network: layers chained together

A neural network is one or more layers chained together: the first
layer's output becomes the second layer's input, and so on, until a
final layer produces the network's overall output. `infer-c`'s network
has exactly two layers:

```
784 inputs → [Layer 1: 128 neurons] → [Layer 2: 10 neurons] → 10 outputs
```

This specific shape — input size, then one hidden layer, then output
size — is called a **multi-layer perceptron**, or **MLP**. It's one of
the simplest possible neural network architectures, and the one this
entire project is built around. The "784 → 128 → 10" shorthand you'll
see throughout this project's documentation describes exactly this: 784
inputs, a hidden layer of 128 neurons, an output layer of 10 neurons (one
per digit class).

### Why "hidden"?

Layer 1 here is called the **hidden layer** because its output isn't
directly the thing we care about — it's an intermediate representation
the network builds up internally, on its way to producing the final
10-number output. Only the *first* layer (which sees the raw pixel
input) and the *last* layer (which produces the final classification)
have an obvious, human-readable meaning. What the hidden layer's 128
numbers actually represent is something the network discovers during
training, not something a human designed by hand — this is part of what
makes neural networks powerful (they find their own useful intermediate
representations) and part of what makes them hard to fully interpret.

---

## Part 4 — Why you need nonlinearity: activation functions

### The problem with only weighted sums

Here's a fact that might be surprising at first: if you chain together
several layers that *only* do weighted-sum-plus-bias, with nothing else,
the entire chain collapses mathematically into something equivalent to a
*single* layer. Stacking linear operations on top of linear operations
just gives you another linear operation — no matter how many layers you
add, you haven't gained any new expressive power.

This is why every layer (except sometimes the very last) applies an
**activation function** after the weighted sum: a simple, deliberately
nonlinear function applied to each neuron's output, one number at a
time. This nonlinearity is what allows a multi-layer network to represent
far more complex patterns than a single layer could.

### ReLU: the activation function this project uses

`infer-c`'s hidden layer uses **ReLU** (Rectified Linear Unit), one of
the simplest and most widely used activation functions in modern neural
networks:

```
ReLU(x) = max(0, x)
```

In words: if the number is negative, replace it with zero; if it's
already zero or positive, leave it unchanged. That's the entire
function — and despite being almost trivially simple, it works
remarkably well in practice and trains faster than many historically
popular alternatives (like the sigmoid function). `infer-c`'s `nn_relu`
function implements exactly this, looping over every value in the
hidden layer's output and zeroing out the negative ones.

### Why "rectified linear"?

"Linear" because for positive inputs, it's just the identity function (no
change). "Rectified" borrows electrical-engineering terminology for
clipping a signal to only allow one direction through — here, clipping
away anything negative. The name is more historical than illuminating;
what matters is the simple `max(0, x)` behavior.

---

## Part 5 — Putting it together: the forward pass

### What "forward pass" means

The **forward pass** is the complete process of taking an input (here,
784 pixel values) and pushing it through every layer of the network, in
order, to produce a final output. "Forward" distinguishes this from
**backpropagation** (covered in Part 7), which moves information
*backward* through the network during training. Inference — actually
using a trained model to make a prediction, which is all `infer-c` does
— is just running the forward pass once per input.

### The forward pass, step by step, for this project's architecture

1. **Input**: 784 normalized pixel values (one flattened, normalized
   MNIST image).
2. **Layer 1 (linear)**: multiply the input by Layer 1's weight matrix,
   add Layer 1's bias — `infer-c`'s `mat_mul` followed by
   `mat_add_bias`. Result: 128 numbers.
3. **Layer 1 (activation)**: apply ReLU to all 128 numbers —
   `nn_relu`. Negative values become 0; positive values are unchanged.
4. **Layer 2 (linear)**: multiply the (now-ReLU'd) 128 numbers by Layer
   2's weight matrix, add Layer 2's bias. Result: 10 numbers, one per
   digit class. These 10 raw numbers are called **logits** — more on
   this term in Part 6.
5. **Layer 2 (activation)**: apply **softmax** (Part 6) to the 10
   logits, turning them into 10 probabilities that sum to 1.

That entire sequence is what `infer-c`'s `nn_forward` function executes,
mechanically, every time you run `bin/infer` on an image. There is no
"intelligence" happening at runtime in any mystical sense — it's
arithmetic, executed in a fixed sequence, using numbers (the weights and
biases) that were determined ahead of time by training.

### Where the weights and biases come from

This is worth being explicit about, because it's the crux of this
project's entire design (see ADR-003 in `docs/adr/` for the original
reasoning): the weights and biases are *not* computed by `infer-c`
itself. They are produced separately, by a training process written in
Python (`python/train.py`), and saved to a file (`weights.bin`) that
`infer-c`'s C code reads. `infer-c` only ever *uses* already-determined
weights — it never learns or adjusts them. This split (train in Python,
infer in C) is deliberate and is explained further in Part 8.
## Part 6 — Softmax: turning numbers into probabilities

### The problem softmax solves

After Layer 2's weighted sum, you have 10 raw numbers — the **logits**
mentioned above. These could be anything: large, small, negative,
positive, with no particular relationship to each other. But what we
actually want is a *probability distribution*: 10 numbers that are each
between 0 and 1, and that all sum to exactly 1, representing "how
confident is the network that this image is a 0, a 1, a 2, ..., a 9."

**Softmax** is the function that converts an arbitrary list of numbers
into exactly that kind of probability distribution. Its formula, for a
list of numbers `x_1, ..., x_n`:

```
softmax(x_i) = exp(x_i) / (exp(x_1) + exp(x_2) + ... + exp(x_n))
```

Where `exp` is the exponential function (`e^x`). In words: exponentiate
every number (which makes everything positive, since `exp` of any real
number is positive), then divide each exponentiated value by the sum of
all of them. This guarantees: every output is positive, and they all sum
to 1 — exactly the two properties a probability distribution needs.

### Why exponentiate at all, instead of just normalizing directly?

You might wonder why not just do `x_i / (x_1 + ... + x_n)` directly,
skipping the exponential. Two reasons: first, the raw logits can be
negative, and dividing by a sum that might itself be near zero or
negative would not reliably produce valid probabilities. Second,
exponentiation has a useful side effect: it *exaggerates* differences.
If one logit is meaningfully larger than the others, exponentiation
makes its share of the final probability disproportionately larger too
— which matches the intuition that a confident network should output
one high probability and several low ones, not ten nearly-equal values.

### The numerical stability trap

Here's a subtlety that `infer-c`'s code handles carefully, and that's
worth understanding because it's a classic, easy-to-miss bug: computing
`exp(x_i)` directly, for an `x_i` that's even moderately large (say,
1000), produces a number so large it **overflows** — exceeds the largest
value a `float` can represent, becoming `inf` (infinity). Once that
happens, the division step produces `inf / inf`, which is mathematically
undefined and shows up as `NaN` ("Not a Number") — a value that silently
poisons every later calculation that touches it.

The fix, which `infer-c`'s `nn_softmax` function implements, is
mathematically simple but easy to overlook: subtract the largest logit
from every logit *before* exponentiating:

```
softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

This produces the *exact same result* mathematically (you can verify
algebraically that subtracting a constant from every term before this
particular division cancels out), but now the largest exponent computed
is always `exp(0) = 1`, and every other exponent is `exp(negative
number)`, which shrinks toward zero instead of blowing up toward
infinity. This is a standard technique used in essentially every
production-grade softmax implementation, not a quirk specific to this
project — but it's also exactly the kind of detail that's invisible
until you hit it, which is why this project's tests specifically include
a case (`test_softmax_large_magnitude_stable`) that deliberately uses a
large input value, to prove the stable version is actually being used
and actually works.

### Confidence

Once you have a softmax output — 10 probabilities summing to 1 — two
more things follow naturally:

- **Predicted class**: whichever of the 10 classes has the *highest*
  probability. Finding this is called taking the **argmax** ("argument
  of the maximum") — not the maximum value itself, but the *position*
  (here, which digit, 0–9) where that maximum occurs.
- **Confidence**: the probability value of the predicted class itself.
  If the network outputs `[0.01, 0.02, 0.01, 0.90, ...]` for digits
  `[0, 1, 2, 3, ...]`, the predicted class is 3, and the confidence is
  0.90 (90%).

This is exactly what `infer-c`'s public API returns: `InferResult`'s
`predicted_class` (the argmax) and `confidence` (the probability at that
position). See `docs/adr/ADR-005...` and the public API design
discussion in this project's ADRs for the explicit reasoning behind
defining confidence this way, rather than, say, using the raw logit
value (which wouldn't have a fixed, comparable scale the way a
probability does).

---

## Part 7 — Training: how the weights get learned (briefly — this part is Python's job, not infer-c's)

`infer-c` never trains anything — but understanding roughly what training
*is* makes clear why the weights it loads aren't arbitrary, and why the
Python/C split (Part 8) makes sense.

### Loss: measuring how wrong the network is

Training needs a way to measure "how wrong was this prediction?" — a
single number that's large when the network's output is far from the
correct answer and small (ideally zero) when it's close. This measure is
called the **loss function**. For classification problems like this one,
a common choice is **cross-entropy loss**, which specifically measures
how different the network's predicted probability distribution is from
the "true" distribution (which, for a labeled example like "this image
is a 7," is just 100% probability on class 7 and 0% on every other
class).

### Gradient descent and backpropagation, in one paragraph

Training works by repeatedly: (1) running the forward pass on some
training examples, (2) computing the loss, (3) figuring out, for *every
single weight and bias in the network*, which direction (increase or
decrease) and by how much that specific number should change to make the
loss slightly smaller, and (4) nudging every parameter slightly in that
direction. Step (3) — calculating how much each of potentially millions
of parameters individually contributed to the error — is done by an
algorithm called **backpropagation**, which uses calculus (the chain
rule) to efficiently work backward from the loss, layer by layer, back
toward the input. Step (4), repeatedly nudging parameters in the
direction that reduces loss, is called **gradient descent**. This entire
loop — forward pass, measure loss, backpropagate, nudge weights, repeat,
often thousands of times over many passes through the training data
(each full pass is called an **epoch**) — is what "training a neural
network" concretely means.

`infer-c`'s `python/train.py` does exactly this, using PyTorch, which
implements backpropagation and gradient descent for you (this is most of
what a deep learning framework like PyTorch, TensorFlow, or JAX actually
provides — automatic, efficient backpropagation, so you don't have to
hand-derive calculus for every architecture). After 10 epochs of this
process on the MNIST training set, `train.py` arrives at a specific set
of weights and biases that, empirically, classify the test set correctly
92.76% of the time — and that's the exact set of numbers saved to
`weights.bin` and loaded by every C function described in Part 5.

### Why this matters even though infer-c doesn't do it

Nothing in `infer-c`'s C code could make sense without this context: the
weight matrices it loads aren't designed by a person, hand-picked, or
derived from some formula — they're the *output* of this training
process, a large set of numbers gradient descent converged on after
seeing thousands of labeled examples. `infer-c`'s job is much narrower
and more mechanical than training: take those already-determined numbers
and run the forward pass (Part 5) as fast and correctly as possible.

---

## Part 8 — Why this project is split into Python (training) and C (inference)

This split is documented formally in this project's `docs/adr/` (see the
ADR establishing the train/infer separation), but the conceptual reason
is worth stating plainly:

**Training** needs: automatic differentiation (backpropagation),
flexibility to experiment with architectures and hyperparameters quickly,
and access to a rich ecosystem of tooling (data loading, visualization,
pretrained models, GPU acceleration). Python, with frameworks like
PyTorch, is extremely well-suited to this — quick to write, quick to
iterate, with backpropagation handled automatically.

**Inference** — once training is done and you just need to *run* a fixed,
already-trained model, repeatedly, as fast as possible, with minimal
memory and zero unnecessary dependencies — has different priorities
entirely: speed, small footprint, predictable behavior, and (for a
project like this one) the educational and portfolio value of
implementing the actual arithmetic by hand, in a systems language, rather
than calling a library that does it for you.

This is also why `infer-c` deliberately does **not** use any ML framework
on the C side, even though plenty exist (TensorFlow Lite, ONNX Runtime,
and others solve exactly this "run a trained model efficiently" problem).
The point of this project is to implement that machinery from scratch —
the matrix multiplication, the activation functions, the file format for
storing trained weights, all of it — as a demonstration of systems
programming, not because no other solution exists.

---

## Part 9 — Mapping every concept to the actual code

A quick-reference table connecting every concept above to where it
concretely lives in this project:

| Concept | Where it lives |
|---|---|
| Flattened, normalized pixel input | `src/data.c`'s `data_load_images` |
| Weight matrix, bias vector | `src/matrix.h`'s `Matrix` type |
| Matrix multiplication (the core of a layer's weighted sum) | `src/matrix.c`'s `mat_mul` |
| Adding the bias | `src/matrix.c`'s `mat_add_bias` |
| ReLU activation | `src/nn.c`'s `nn_relu` |
| Softmax activation (numerically stable) | `src/nn.c`'s `nn_softmax` |
| The full forward pass (all layers, in sequence) | `src/nn.c`'s `nn_forward` |
| Loading trained weights from disk | `src/model.c`'s `model_load` |
| The public, stable API a user actually calls | `src/infer_c.h`'s `infer_model_load` / `infer_run` |
| Predicted class (argmax) and confidence | `InferResult.predicted_class` / `InferResult.confidence` |
| Training (forward pass + loss + backprop + gradient descent) | `python/train.py`, using PyTorch |
| Saving trained weights to the file C reads | `python/export_weights.py` |
| The single-image command-line demo | `src/main.c` |
| Measuring accuracy across the full test set | `src/benchmark.c` |

---

## Part 10 — Glossary

A quick lookup for every term used in this guide.

- **Neuron / unit**: the smallest computational element — a weighted sum
  of inputs, plus a bias.
- **Weight**: a learned number controlling how strongly one input
  influences one neuron's output.
- **Bias**: a learned constant added to a neuron's weighted sum.
- **Layer**: a group of neurons that each independently process the same
  input.
- **MLP (multi-layer perceptron)**: a neural network made of one or more
  layers, each fully connected to the next, with nonlinear activations in
  between. The architecture this project implements.
- **Hidden layer**: any layer that isn't the first (input) or last
  (output) layer — its output isn't directly interpretable, just an
  internal representation the network builds.
- **Activation function**: a nonlinear function applied to a layer's
  output, necessary to let stacked layers represent more than a stacked
  linear layer could alone.
- **ReLU**: `max(0, x)` — the activation function used in this project's
  hidden layer.
- **Logits**: the raw, un-normalized output of a network's final linear
  layer, before any activation like softmax is applied.
- **Softmax**: a function converting a list of arbitrary numbers into a
  valid probability distribution (all positive, summing to 1).
- **Argmax**: the position/index of the largest value in a list — used
  here to turn a probability distribution into a single predicted class.
- **Confidence**: in this project, the probability value (post-softmax)
  of the predicted class.
- **Forward pass**: running an input through every layer of a network, in
  order, to produce an output. What "inference" means in practice.
- **Inference**: using an already-trained model to make a prediction
  (as opposed to training it).
- **Loss function**: a single number measuring how wrong a prediction
  was; what training tries to minimize.
- **Cross-entropy loss**: a specific loss function commonly used for
  classification, measuring the difference between a predicted
  probability distribution and the true one.
- **Gradient descent**: the algorithm of repeatedly nudging a network's
  parameters in the direction that reduces the loss.
- **Backpropagation**: the algorithm (based on the calculus chain rule)
  that efficiently computes how much each individual parameter
  contributed to the loss, enabling gradient descent.
- **Epoch**: one complete pass through the entire training dataset during
  training.
- **Training**: the overall process (forward pass, loss, backpropagation,
  gradient descent, repeated many times) of determining a network's
  weights and biases from labeled example data.
- **Classification**: a prediction task where the output is one of a
  fixed set of categories (as opposed to regression, a continuous value).
- **Normalization (of input data)**: rescaling input values into a small,
  consistent range (here, pixel values from [0, 255] to [0.0, 1.0])
  before feeding them into a network.
- **Matrix**: a grid of numbers; the natural way to represent a layer's
  entire set of weights, and the structure `mat_mul` operates on.
- **Overflow (numerical)**: when a calculation produces a value too large
  for its data type to represent, often resulting in `inf` or wrapping to
  an incorrect value.
- **NaN ("Not a Number")**: a special floating-point value representing
  an undefined result (like `0/0` or `inf/inf`), which silently
  propagates through further calculations if not caught.

---

## Where to go next

If you want to keep exploring from here:

- Read `docs/adr/` in this repository — the Architecture Decision Records
  document not just *what* this project does but *why*, including
  several genuinely subtle engineering decisions (numerical stability,
  memory ownership, file format design, public API error handling) that
  this guide only summarizes.
- Try modifying `python/train.py` — change the hidden layer size, the
  number of epochs, or the learning rate, and see how accuracy changes.
  This is the fastest way to build intuition for what these
  hyperparameters actually do.
- For the natural next step beyond an MLP on a small, simple dataset:
  look into **convolutional neural networks** (CNNs), which are the
  standard architecture for image tasks at any real scale, and were
  deliberately left out of this project's scope (see the README's
  "Explicitly out of scope" section).
- If your interest is specifically in NLP (which, again, this project
  does not cover): the natural starting point from here is learning how
  text gets converted into numbers in the first place — **tokenization**
  and **word/token embeddings** — since everything from Part 3 onward
  (layers, activations, training) applies just as directly to text-based
  networks once the input is in numeric form.
