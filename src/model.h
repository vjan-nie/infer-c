#ifndef INFER_C_MODEL_H
#define INFER_C_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "matrix.h"

typedef struct {
    Matrix weight;          /* in_features x out_features (transposed from file) */
    Matrix bias;             /* 1 x out_features */
    uint32_t activation_id;  /* 0=none, 1=relu, 2=softmax, per ADR-004 */
} Layer;

typedef struct {
    Layer* layers;
    int n_layers;
} Model;

/* Loads a model from a weight file (ADR-004/ADR-010 format). Verifies
 * the file's CRC32 before interpreting any structured field. Returns
 * NULL on any environment error: file not found, read error, checksum
 * mismatch, bad magic/version, allocation failure, or internally
 * inconsistent layer dimensions despite a valid checksum. */
Model* model_load(const char* path);

/* Frees every layer's weight/bias matrices, the layers array, and the
 * Model struct itself (model_load heap-allocates it). m must not be
 * used after this call. */
void model_free(Model* m);

/* Standard CRC-32/ISO-HDLC (poly 0xEDB88320 reflected, init 0xFFFFFFFF,
 * final XOR 0xFFFFFFFF), table-driven. Exposed (non-static) so it is
 * directly testable per ADR-007, not because it is part of the
 * intended public API. */
uint32_t model_crc32(const unsigned char* data, size_t len);

/* Pure predicate, never aborts: true iff every consecutive layer pair's
 * out_features (layers[i].weight.cols) matches the next layer's
 * in_features (layers[i+1].weight.rows). */
int model_layers_chain_compatible(const Model* model);

#endif
