#ifndef INFER_C_MATRIX_H
#define INFER_C_MATRIX_H

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

/* Allocates data with malloc. Caller owns the result and must call
 * mat_free. rows and cols must be positive — enforced by assert, a
 * programmer error per ADR-005. On allocation failure for a validly-
 * sized request, returns a Matrix with data == NULL; caller must
 * check. */
Matrix mat_create(int rows, int cols);

/* Frees m->data, sets data = NULL, rows = cols = 0. Safe to call twice. */
void mat_free(Matrix* m);

/* Pure, side-effect-free dimension checks. Never abort. Used internally
 * (before assert) and directly by tests. */
int mat_dims_valid(int rows, int cols);
int mat_dims_compatible_for_mul(const Matrix* a, const Matrix* b);
int mat_dims_compatible_for_add_bias(const Matrix* m, const Matrix* bias);

/* out must already be created with dimensions a->rows x b->cols. */
void mat_mul(const Matrix* a, const Matrix* b, Matrix* out);

/* In-place: adds bias (1 x m->cols row vector) to every row of m. */
void mat_add_bias(Matrix* m, const Matrix* bias);

/* dst must already be created with the same dimensions as src. */
void mat_copy(const Matrix* src, Matrix* dst);

#endif
