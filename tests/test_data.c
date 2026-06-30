#include "../src/data.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_PATH "/tmp/test_data_fixture.idx"
#define PIXELS_PER_IMAGE 784

static void write_be_u32(FILE *f, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v >> 24);
    b[1] = (unsigned char)(v >> 16);
    b[2] = (unsigned char)(v >>  8);
    b[3] = (unsigned char)v;
    assert(fwrite(b, 1, (size_t)4, f) == (size_t)4);
}

static void write_image_fixture(uint32_t magic, uint32_t n,
                                uint32_t rows, uint32_t cols,
                                const unsigned char *pixels, size_t pixel_bytes) {
    FILE *f = fopen(FIXTURE_PATH, "wb");
    assert(f != NULL);
    write_be_u32(f, magic);
    write_be_u32(f, n);
    write_be_u32(f, rows);
    write_be_u32(f, cols);
    if (pixels && pixel_bytes > 0) {
        assert(fwrite(pixels, 1, pixel_bytes, f) == pixel_bytes);
    }
    fclose(f);
}

static void write_label_fixture(uint32_t magic, uint32_t n,
                                const unsigned char *labels, size_t label_bytes) {
    FILE *f = fopen(FIXTURE_PATH, "wb");
    assert(f != NULL);
    write_be_u32(f, magic);
    write_be_u32(f, n);
    if (labels && label_bytes > 0) {
        assert(fwrite(labels, 1, label_bytes, f) == label_bytes);
    }
    fclose(f);
}

/* --- Image tests --- */

static void test_load_images_all_zeros(void) {
    unsigned char pixels[PIXELS_PER_IMAGE];
    memset(pixels, 0x00, sizeof(pixels));
    write_image_fixture(0x00000803U, 1, 28, 28, pixels, sizeof(pixels));

    int n = 0;
    float *result = data_load_images(FIXTURE_PATH, &n);
    assert(result != NULL);
    assert(n == 1);
    for (int i = 0; i < PIXELS_PER_IMAGE; i++) {
        assert(result[i] == 0.0f);
    }
    free(result);
    remove(FIXTURE_PATH);
}

static void test_load_images_all_255(void) {
    unsigned char pixels[PIXELS_PER_IMAGE];
    memset(pixels, 0xFF, sizeof(pixels));
    write_image_fixture(0x00000803U, 1, 28, 28, pixels, sizeof(pixels));

    int n = 0;
    float *result = data_load_images(FIXTURE_PATH, &n);
    assert(result != NULL);
    assert(n == 1);
    for (int i = 0; i < PIXELS_PER_IMAGE; i++) {
        assert(result[i] == 1.0f);
    }
    free(result);
    remove(FIXTURE_PATH);
}

static void test_load_images_known_pixel(void) {
    unsigned char pixels[PIXELS_PER_IMAGE];
    memset(pixels, 0, sizeof(pixels));
    pixels[0] = 128;
    write_image_fixture(0x00000803U, 1, 28, 28, pixels, sizeof(pixels));

    int n = 0;
    float *result = data_load_images(FIXTURE_PATH, &n);
    assert(result != NULL);
    assert(n == 1);
    assert(fabsf(result[0] - 128.0f / 255.0f) < 1e-6f);
    for (int i = 1; i < PIXELS_PER_IMAGE; i++) {
        assert(result[i] == 0.0f);
    }
    free(result);
    remove(FIXTURE_PATH);
}

static void test_load_images_multi(void) {
    unsigned char pixels[3 * PIXELS_PER_IMAGE];
    memset(pixels, 0, sizeof(pixels));
    pixels[PIXELS_PER_IMAGE] = 200;  /* img[1][0] = 200 */
    write_image_fixture(0x00000803U, 3, 28, 28, pixels, sizeof(pixels));

    int n = 0;
    float *result = data_load_images(FIXTURE_PATH, &n);
    assert(result != NULL);
    assert(n == 3);
    assert(fabsf(result[PIXELS_PER_IMAGE] - 200.0f / 255.0f) < 1e-6f);
    free(result);
    remove(FIXTURE_PATH);
}

static void test_load_images_bad_magic(void) {
    unsigned char pixels[PIXELS_PER_IMAGE];
    memset(pixels, 0, sizeof(pixels));
    write_image_fixture(0x00000801U, 1, 28, 28, pixels, sizeof(pixels));

    int n = 0;
    float *result = data_load_images(FIXTURE_PATH, &n);
    assert(result == NULL);
    remove(FIXTURE_PATH);
}

static void test_load_images_bad_dims(void) {
    unsigned char pixels[27 * 28];
    memset(pixels, 0, sizeof(pixels));
    write_image_fixture(0x00000803U, 1, 27, 28, pixels, sizeof(pixels));

    int n = 0;
    float *result = data_load_images(FIXTURE_PATH, &n);
    assert(result == NULL);
    remove(FIXTURE_PATH);
}

static void test_load_images_truncated(void) {
    /* Header declares n=100 but only one image's worth of pixels follows */
    unsigned char pixels[PIXELS_PER_IMAGE];
    memset(pixels, 0, sizeof(pixels));
    write_image_fixture(0x00000803U, 100, 28, 28, pixels, sizeof(pixels));

    int n = 0;
    float *result = data_load_images(FIXTURE_PATH, &n);
    assert(result == NULL);
    remove(FIXTURE_PATH);
}

/* --- Label tests --- */

static void test_load_labels_valid(void) {
    unsigned char labels[3] = {5, 0, 9};
    write_label_fixture(0x00000801U, 3, labels, 3);

    int n = 0;
    uint8_t *result = data_load_labels(FIXTURE_PATH, &n);
    assert(result != NULL);
    assert(n == 3);
    assert(result[0] == 5);
    assert(result[1] == 0);
    assert(result[2] == 9);
    free(result);
    remove(FIXTURE_PATH);
}

static void test_load_labels_bad_magic(void) {
    unsigned char labels[3] = {1, 2, 3};
    write_label_fixture(0x00000803U, 3, labels, 3);

    int n = 0;
    uint8_t *result = data_load_labels(FIXTURE_PATH, &n);
    assert(result == NULL);
    remove(FIXTURE_PATH);
}

static void test_load_labels_truncated(void) {
    /* Header declares n=50 but only 1 byte follows */
    unsigned char label = 7;
    write_label_fixture(0x00000801U, 50, &label, 1);

    int n = 0;
    uint8_t *result = data_load_labels(FIXTURE_PATH, &n);
    assert(result == NULL);
    remove(FIXTURE_PATH);
}

int main(void) {
    test_load_images_all_zeros();
    printf("test_load_images_all_zeros: PASS\n");
    test_load_images_all_255();
    printf("test_load_images_all_255: PASS\n");
    test_load_images_known_pixel();
    printf("test_load_images_known_pixel: PASS\n");
    test_load_images_multi();
    printf("test_load_images_multi: PASS\n");
    test_load_images_bad_magic();
    printf("test_load_images_bad_magic: PASS\n");
    test_load_images_bad_dims();
    printf("test_load_images_bad_dims: PASS\n");
    test_load_images_truncated();
    printf("test_load_images_truncated: PASS\n");
    test_load_labels_valid();
    printf("test_load_labels_valid: PASS\n");
    test_load_labels_bad_magic();
    printf("test_load_labels_bad_magic: PASS\n");
    test_load_labels_truncated();
    printf("test_load_labels_truncated: PASS\n");

    printf("All data tests passed.\n");
    return 0;
}
