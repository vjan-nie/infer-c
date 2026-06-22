#ifndef INFER_C_NN_H
#define INFER_C_NN_H

#include "matrix.h"
#include "model.h"

typedef struct {
    Matrix logits; /* 1 x n_classes, pre-softmax */
    Matrix probs;  /* 1 x n_classes, post-softmax */
} ForwardResult;

/* input must be 1 x (model's first layer in_features). out->logits and
 * out->probs must already be created by the caller as
 * 1 x (model's last layer out_features) — same pre-allocation
 * convention as mat_mul's out. Intermediate per-layer activations are
 * allocated and freed internally. */
void nn_forward(const Model* model, const Matrix* input, ForwardResult* out);

/* In place: m->data[i] = max(0, m->data[i]). */
void nn_relu(Matrix* m);

/* out must already be created with in's dimensions. Numerically
 * stable: subtracts max(in) before exponentiating, never exp(x_i)
 * directly. */
void nn_softmax(const Matrix* in, Matrix* out);

#endif
