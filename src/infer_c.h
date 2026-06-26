#ifndef INFER_C_H
#define INFER_C_H

/* Public API of the inference engine (ADR-001): the only header an
 * external consumer of this library needs to include. */

/* Opaque handle to a loaded model. */
typedef struct InferModel InferModel;

/* Loads a model from the file at model_path. Returns NULL on any
 * loading failure (file not found, unreadable, corrupted, or otherwise
 * invalid). The returned handle must be freed exactly once with
 * infer_model_free. */
InferModel* infer_model_load(const char* model_path);

/* Releases every resource owned by model. model must not be used after
 * this call. Safe to call with model == NULL (no-op). Not safe to call
 * twice on the same handle. */
void infer_model_free(InferModel* model);

/* Result of a single classification run. */
typedef struct {
    int predicted_class;  /* index of the most likely class, in [0, n) */
    float confidence;     /* probability assigned to predicted_class, in [0, 1] */
} InferResult;

/* Classifies input, an array of input_size floats, using model.
 *
 * Returns { .predicted_class = -1, .confidence = 0.0f } if model is
 * NULL, if input_size does not match the number of values this model
 * expects, or if a result could not otherwise be computed. Callers
 * must check predicted_class >= 0 before trusting confidence. */
InferResult infer_run(const InferModel* model, const float* input, int input_size);

#endif
