#include "nn.h"

#include <math.h>

void nn_relu(Matrix* m) {
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) {
        if (m->data[i] < 0.0f) {
            m->data[i] = 0.0f;
        }
    }
}

void nn_softmax(const Matrix* in, Matrix* out) {
    int n = in->rows * in->cols;

    float max_val = in->data[0];
    for (int i = 1; i < n; i++) {
        if (in->data[i] > max_val) {
            max_val = in->data[i];
        }
    }

    /* exp(x_i - max(x)) instead of exp(x_i): mathematically identical
     * result, but prevents overflow on large-magnitude logits. Safe to
     * alias out == in: max_val is computed before any write, and each
     * out->data[i] is written only after in->data[i] has been read. */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = expf(in->data[i] - max_val);
        out->data[i] = e;
        sum += e;
    }

    for (int i = 0; i < n; i++) {
        out->data[i] /= sum;
    }
}

void nn_forward(const Model* model, const Matrix* input, ForwardResult* out) {
    const Matrix* current = input;
    Matrix prev_owned = {0};
    int has_prev_owned = 0;

    for (int i = 0; i < model->n_layers; i++) {
        const Layer* layer = &model->layers[i];
        Matrix tmp = mat_create(1, layer->weight.cols);
        mat_mul(current, &layer->weight, &tmp);
        mat_add_bias(&tmp, &layer->bias);

        if (has_prev_owned) {
            mat_free(&prev_owned);
        }

        if (i == model->n_layers - 1) {
            mat_copy(&tmp, &out->logits);
            nn_softmax(&out->logits, &out->probs);
            mat_free(&tmp);
            return;
        }

        if (layer->activation_id == 1) {
            nn_relu(&tmp);
        } else if (layer->activation_id == 2) {
            nn_softmax(&tmp, &tmp);
        }
        /* activation_id == 0: no-op, defined in the format but unused
         * by the current architecture. */

        prev_owned = tmp;
        has_prev_owned = 1;
        current = &prev_owned;
    }
}
