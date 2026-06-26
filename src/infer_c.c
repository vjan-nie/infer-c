#include "infer_c.h"

#include "matrix.h"
#include "model.h"
#include "nn.h"

#include <stdlib.h>

struct InferModel {
    Model* model;
};

InferModel* infer_model_load(const char* model_path) {
    Model* model = model_load(model_path);
    if (model == NULL) {
        return NULL;
    }

    InferModel* wrapper = (InferModel*)malloc(sizeof(InferModel));
    if (wrapper == NULL) {
        model_free(model);
        return NULL;
    }

    wrapper->model = model;
    return wrapper;
}

void infer_model_free(InferModel* model) {
    if (model == NULL) {
        return;
    }
    model_free(model->model);
    free(model);
}

InferResult infer_run(const InferModel* model, const float* input, int input_size) {
    InferResult error_result = { -1, 0.0f };

    if (model == NULL) {
        return error_result;
    }

    Model* m = model->model;
    int expected_input_size = m->layers[0].weight.rows;
    if (input_size != expected_input_size) {
        return error_result;
    }

    int n_classes = m->layers[m->n_layers - 1].weight.cols;
    Matrix in = { (float*)input, 1, input_size };

    ForwardResult fwd;
    fwd.logits = mat_create(1, n_classes);
    fwd.probs = mat_create(1, n_classes);
    if (fwd.logits.data == NULL || fwd.probs.data == NULL) {
        mat_free(&fwd.logits);
        mat_free(&fwd.probs);
        return error_result;
    }

    nn_forward(m, &in, &fwd);

    int predicted_class = 0;
    float best = fwd.probs.data[0];
    for (int i = 1; i < n_classes; i++) {
        if (fwd.probs.data[i] > best) {
            best = fwd.probs.data[i];
            predicted_class = i;
        }
    }

    InferResult result = { predicted_class, best };

    mat_free(&fwd.logits);
    mat_free(&fwd.probs);
    return result;
}
