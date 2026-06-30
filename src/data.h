#ifndef DATA_H
#define DATA_H

#include <stdint.h>

/*
 * Load an IDX image file. Returns a heap-allocated array of
 * n_images * 784 floats, pre-normalized to [0.0, 1.0].
 * Sets *n_out to the number of images on success.
 * Returns NULL on any error (file not found, bad magic, wrong dimensions,
 * truncated file, or allocation failure). Caller must free() the result.
 */
float *data_load_images(const char *path, int *n_out);

/*
 * Load an IDX label file. Returns a heap-allocated array of *n_out uint8_t labels.
 * Returns NULL on any error (file not found, bad magic, truncated file, or
 * allocation failure). Caller must free() the result.
 */
uint8_t *data_load_labels(const char *path, int *n_out);

#endif /* DATA_H */
