#include "infer_c.h"
#include "data.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
	if (argc != 3) {
		fprintf(stderr, "Usage: benchmark <images.idx> <labels.idx>\n");
		return 1;
	}

	int n_images = 0;
	float *images = data_load_images(argv[1], &n_images);
	if (!images) {
		fprintf(stderr, "Error: failed to load images '%s'\n", argv[1]);
		return 1;
	}

	int n_labels = 0;
	uint8_t *labels = data_load_labels(argv[2], &n_labels);
	if (!labels) {
		fprintf(stderr, "Error: failed to load labels '%s'\n", argv[2]);
		free(images);
		return 1;
	}

	if (n_images != n_labels) {
		fprintf(stderr, "Error: image/label count mismatch: %d vs %d\n",
				n_images, n_labels);
		free(images);
		free(labels);
		return 1;
	}

	InferModel *model = infer_model_load("weights.bin");
	if (!model) {
		fprintf(stderr, "Error: failed to load model 'weights.bin'\n");
		free(images);
		free(labels);
		return 1;
	}

	int correct = 0;
	for (int i = 0; i < n_images; i++) {
		InferResult result = infer_run(model, images + (size_t)i * 784, 784);
		if (result.predicted_class < 0) {
			fprintf(stderr, "Warning: inference failed on image %d\n", i);
			continue;
		}
		if (result.predicted_class == (int)labels[i]) {
			correct++;
		}
	}

	if (n_images > 0) {
		printf("Accuracy: %d / %d (%.2f%%)\n",
			correct, n_images,
			(double)(100.0f * (float)correct / (float)n_images));
	} else {
		printf("Accuracy: %d / %d (no images to evaluate)\n", correct, n_images);
	}
	free(images);
	free(labels);
	infer_model_free(model);
	return 0;
}
