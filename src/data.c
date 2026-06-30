#include "data.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define IDX_MAGIC_IMAGES 0x00000803U
#define IDX_MAGIC_LABELS 0x00000801U

#define IDX_IMAGE_HEADER_BYTES 16
#define IDX_LABEL_HEADER_BYTES  8

/* ADR-013: validate against this project's fixed input size, not just IDX
 * format validity. The trained model expects exactly 28x28 = 784 inputs. */
#define MNIST_ROWS   28U
#define MNIST_COLS   28U
#define MNIST_PIXELS 784U  /* MNIST_ROWS * MNIST_COLS */

/* IDX files are big-endian; x86 is little-endian. Read four bytes MSB-first
 * explicitly so the result is correct on both endian hosts. Do not use memcpy
 * here — that would silently assume host byte order (as model.c does for its
 * own little-endian format). These two conventions must not blur. */
static uint32_t read_be_u32(const unsigned char *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

float *data_load_images(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long file_size = ftell(f);
    if (file_size < IDX_IMAGE_HEADER_BYTES) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    unsigned char *buf = malloc((size_t)file_size);
    if (!buf) { fclose(f); return NULL; }

    if (fread(buf, 1, (size_t)file_size, f) != (size_t)file_size) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);

    uint32_t magic = read_be_u32(buf);
    uint32_t n     = read_be_u32(buf + 4);
    uint32_t rows  = read_be_u32(buf + 8);
    uint32_t cols  = read_be_u32(buf + 12);

    if (magic != IDX_MAGIC_IMAGES)              { free(buf); return NULL; }
    if (rows != MNIST_ROWS || cols != MNIST_COLS) { free(buf); return NULL; }
    if (n > (uint32_t)(INT_MAX / (int)MNIST_PIXELS)) { free(buf); return NULL; }

    size_t pixel_bytes = (size_t)n * MNIST_PIXELS;
    if ((size_t)file_size - IDX_IMAGE_HEADER_BYTES < pixel_bytes) {
        free(buf); return NULL;
    }

    float *out = malloc(pixel_bytes > 0 ? pixel_bytes * sizeof(float) : sizeof(float));
    if (!out) { free(buf); return NULL; }

    const unsigned char *src = buf + IDX_IMAGE_HEADER_BYTES;
    for (size_t i = 0; i < pixel_bytes; i++) {
        out[i] = (float)src[i] / 255.0f;
    }

    free(buf);
    *n_out = (int)n;
    return out;
}

uint8_t *data_load_labels(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long file_size = ftell(f);
    if (file_size < IDX_LABEL_HEADER_BYTES) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    unsigned char *buf = malloc((size_t)file_size);
    if (!buf) { fclose(f); return NULL; }

    if (fread(buf, 1, (size_t)file_size, f) != (size_t)file_size) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);

    uint32_t magic = read_be_u32(buf);
    uint32_t n     = read_be_u32(buf + 4);

    if (magic != IDX_MAGIC_LABELS)    { free(buf); return NULL; }
    if (n > (uint32_t)INT_MAX)        { free(buf); return NULL; }

    size_t label_bytes = (size_t)n;
    if ((size_t)file_size - IDX_LABEL_HEADER_BYTES < label_bytes) {
        free(buf); return NULL;
    }

    uint8_t *out = malloc(label_bytes > 0 ? label_bytes : 1);
    if (!out) { free(buf); return NULL; }

    const unsigned char *src = buf + IDX_LABEL_HEADER_BYTES;
    for (size_t i = 0; i < label_bytes; i++) {
        out[i] = (uint8_t)src[i];
    }

    free(buf);
    *n_out = (int)n;
    return out;
}
