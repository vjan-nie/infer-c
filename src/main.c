#include "infer_c.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>

/* Print the 28x28 float pixel array as ASCII art.
 * Skips every other row to compensate for ~2:1 terminal character height:width,
 * producing a roughly square 14x28 rendering. */
static void print_digit_art(const float *pixels) {
    for (int row = 0; row < 28; row += 2) {
        for (int col = 0; col < 28; col++)
            putchar(pixels[row * 28 + col] > 0.3f ? '#' : ' ');
        putchar('\n');
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: infer <model.bin> <image.idx>\n");
        return 1;
    }

    InferModel *model = infer_model_load(argv[1]);
    if (!model) {
        fprintf(stderr, "Error: failed to load model '%s'\n", argv[1]);
        return 1;
    }

    int n_images = 0;
    float *images = data_load_images(argv[2], &n_images);
    if (!images) {
        fprintf(stderr, "Error: failed to load image '%s'\n", argv[2]);
        infer_model_free(model);
        return 1;
    }

    if (n_images != 1) {
        fprintf(stderr, "Error: expected a 1-image IDX file, got %d\n", n_images);
        free(images);
        infer_model_free(model);
        return 1;
    }

    print_digit_art(images);

    InferResult result = infer_run(model, images, 784);
    if (result.predicted_class < 0) {
        fprintf(stderr, "Error: inference failed\n");
        free(images);
        infer_model_free(model);
        return 1;
    }

    printf("Predicted digit: %d\n", result.predicted_class);
    printf("Confidence: %.1f%%\n", result.confidence * 100.0f);

    free(images);
    infer_model_free(model);
    return 0;
}
