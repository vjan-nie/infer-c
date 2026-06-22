#include "matrix.h"

#include <assert.h>
#include <stdlib.h>

static inline float mat_at(const Matrix* m, int row, int col) {
    return m->data[row * m->cols + col];
}

static inline void mat_set(Matrix* m, int row, int col, float value) {
    m->data[row * m->cols + col] = value;
}

int mat_dims_valid(int rows, int cols) {
    return rows > 0 && cols > 0;
}

Matrix mat_create(int rows, int cols) {
    assert(mat_dims_valid(rows, cols));

    Matrix m;
    m.data = (float*)malloc((size_t)rows * (size_t)cols * sizeof(float));
    m.rows = rows;
    m.cols = cols;
    return m;
}

void mat_free(Matrix* m) {
    free(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

int mat_dims_compatible_for_mul(const Matrix* a, const Matrix* b) {
    return a->cols == b->rows;
}

int mat_dims_compatible_for_add_bias(const Matrix* m, const Matrix* bias) {
    return bias->rows == 1 && bias->cols == m->cols;
}

void mat_mul(const Matrix* a, const Matrix* b, Matrix* out) {
    assert(mat_dims_compatible_for_mul(a, b));
    assert(out->rows == a->rows && out->cols == b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a->cols; k++) {
                sum += mat_at(a, i, k) * mat_at(b, k, j);
            }
            mat_set(out, i, j, sum);
        }
    }
}

void mat_add_bias(Matrix* m, const Matrix* bias) {
    assert(mat_dims_compatible_for_add_bias(m, bias));

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            mat_set(m, i, j, mat_at(m, i, j) + mat_at(bias, 0, j));
        }
    }
}

void mat_copy(const Matrix* src, Matrix* dst) {
    assert(src->rows == dst->rows && src->cols == dst->cols);

    for (int i = 0; i < src->rows * src->cols; i++) {
        dst->data[i] = src->data[i];
    }
}
