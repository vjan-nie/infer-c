#include "../src/matrix.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>

static void test_create_valid_dims(void) {
    Matrix m = mat_create(2, 3);
    assert(m.data != NULL);
    assert(m.rows == 2);
    assert(m.cols == 3);
    mat_free(&m);
}

static void test_create_alloc_failure_returns_null(void) {
    /* 1e9 x 1e9 floats is ~4 exabytes -- unsatisfiable on any real
     * system, but the malloc size argument stays under 2^63 so it
     * doesn't trip valgrind's "fishy (possibly negative) size"
     * heuristic the way INT_MAX x INT_MAX does. */
    Matrix m = mat_create(1000000000, 1000000000);
    assert(m.data == NULL);
    mat_free(&m);
    assert(m.data == NULL);
}

static void test_dims_valid_positive(void) {
    assert(mat_dims_valid(2, 3) == 1);
}

static void test_dims_valid_zero_rejected(void) {
    assert(mat_dims_valid(0, 3) == 0);
    assert(mat_dims_valid(3, 0) == 0);
}

static void test_dims_valid_negative_rejected(void) {
    assert(mat_dims_valid(-1, 3) == 0);
    assert(mat_dims_valid(3, -1) == 0);
}

static void test_free_then_free_again(void) {
    Matrix m = mat_create(2, 2);
    mat_free(&m);
    assert(m.data == NULL);
    mat_free(&m);
    assert(m.data == NULL);
}

static void test_mul_2x3_times_3x2(void) {
    Matrix a = mat_create(2, 3);
    Matrix b = mat_create(3, 2);
    Matrix out = mat_create(2, 2);

    float a_vals[6] = {1, 2, 3, 4, 5, 6};
    float b_vals[6] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; i++) {
        a.data[i] = a_vals[i];
        b.data[i] = b_vals[i];
    }

    mat_mul(&a, &b, &out);

    assert(out.data[0] == 58.0f);
    assert(out.data[1] == 64.0f);
    assert(out.data[2] == 139.0f);
    assert(out.data[3] == 154.0f);

    mat_free(&a);
    mat_free(&b);
    mat_free(&out);
}

static void test_dims_compatible_for_mul_valid(void) {
    Matrix a = mat_create(2, 3);
    Matrix b = mat_create(3, 4);
    assert(mat_dims_compatible_for_mul(&a, &b) == 1);
    mat_free(&a);
    mat_free(&b);
}

static void test_dims_compatible_for_mul_invalid(void) {
    Matrix a = mat_create(2, 3);
    Matrix b = mat_create(4, 2);
    assert(mat_dims_compatible_for_mul(&a, &b) == 0);
    mat_free(&a);
    mat_free(&b);
}

static void test_add_bias_broadcasts_per_row(void) {
    Matrix m = mat_create(2, 3);
    Matrix bias = mat_create(1, 3);

    float m_vals[6] = {1, 2, 3, 4, 5, 6};
    float bias_vals[3] = {10, 20, 30};
    for (int i = 0; i < 6; i++) m.data[i] = m_vals[i];
    for (int i = 0; i < 3; i++) bias.data[i] = bias_vals[i];

    mat_add_bias(&m, &bias);

    assert(m.data[0] == 11.0f);
    assert(m.data[1] == 22.0f);
    assert(m.data[2] == 33.0f);
    assert(m.data[3] == 14.0f);
    assert(m.data[4] == 25.0f);
    assert(m.data[5] == 36.0f);

    mat_free(&m);
    mat_free(&bias);
}

static void test_dims_compatible_for_add_bias_valid(void) {
    Matrix m = mat_create(2, 3);
    Matrix bias = mat_create(1, 3);
    assert(mat_dims_compatible_for_add_bias(&m, &bias) == 1);
    mat_free(&m);
    mat_free(&bias);
}

static void test_dims_compatible_for_add_bias_invalid(void) {
    Matrix m = mat_create(2, 3);
    Matrix bias = mat_create(1, 4);
    assert(mat_dims_compatible_for_add_bias(&m, &bias) == 0);
    mat_free(&m);
    mat_free(&bias);
}

static void test_copy_is_independent(void) {
    Matrix src = mat_create(2, 2);
    Matrix dst = mat_create(2, 2);

    float src_vals[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; i++) src.data[i] = src_vals[i];

    mat_copy(&src, &dst);
    dst.data[0] = 99.0f;

    assert(src.data[0] == 1.0f);
    assert(dst.data[0] == 99.0f);

    mat_free(&src);
    mat_free(&dst);
}

static void test_edge_case_1x1(void) {
    Matrix a = mat_create(1, 1);
    Matrix b = mat_create(1, 1);
    Matrix out = mat_create(1, 1);
    Matrix copy = mat_create(1, 1);

    a.data[0] = 3.0f;
    b.data[0] = 4.0f;

    mat_mul(&a, &b, &out);
    assert(out.data[0] == 12.0f);

    mat_copy(&a, &copy);
    assert(copy.data[0] == 3.0f);

    mat_free(&a);
    mat_free(&b);
    mat_free(&out);
    mat_free(&copy);
}

static void test_edge_case_row_vector_times_column_vector(void) {
    Matrix row = mat_create(1, 3);
    Matrix col = mat_create(3, 1);
    Matrix out = mat_create(1, 1);

    float row_vals[3] = {1, 2, 3};
    float col_vals[3] = {4, 5, 6};
    for (int i = 0; i < 3; i++) {
        row.data[i] = row_vals[i];
        col.data[i] = col_vals[i];
    }

    mat_mul(&row, &col, &out);
    assert(out.data[0] == 32.0f);

    mat_free(&row);
    mat_free(&col);
    mat_free(&out);
}

int main(void) {
    test_create_valid_dims();
    test_create_alloc_failure_returns_null();
    test_dims_valid_positive();
    test_dims_valid_zero_rejected();
    test_dims_valid_negative_rejected();
    test_free_then_free_again();
    test_mul_2x3_times_3x2();
    test_dims_compatible_for_mul_valid();
    test_dims_compatible_for_mul_invalid();
    test_add_bias_broadcasts_per_row();
    test_dims_compatible_for_add_bias_valid();
    test_dims_compatible_for_add_bias_invalid();
    test_copy_is_independent();
    test_edge_case_1x1();
    test_edge_case_row_vector_times_column_vector();

    printf("All matrix tests passed.\n");
    return 0;
}
