#include "../src/infer_c.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_infer_model_load_valid_file(void) {
    InferModel* model = infer_model_load("weights.bin");
    assert(model != NULL);
    infer_model_free(model);
}

static void test_infer_model_load_missing_file_fails(void) {
    InferModel* model = infer_model_load("bin/does_not_exist.bin");
    assert(model == NULL);
}

static void test_infer_model_load_corrupted_checksum_fails(void) {
    FILE* src = fopen("weights.bin", "rb");
    assert(src != NULL);
    assert(fseek(src, 0, SEEK_END) == 0);
    long size = ftell(src);
    assert(size > 0);
    assert(fseek(src, 0, SEEK_SET) == 0);

    unsigned char* buf = (unsigned char*)malloc((size_t)size);
    assert(buf != NULL);
    assert(fread(buf, 1, (size_t)size, src) == (size_t)size);
    fclose(src);

    buf[100] ^= 0xFFU;

    const char* corrupted_path = "bin/corrupted_weights_infer_c.bin";
    FILE* dst = fopen(corrupted_path, "wb");
    assert(dst != NULL);
    assert(fwrite(buf, 1, (size_t)size, dst) == (size_t)size);
    fclose(dst);
    free(buf);

    InferModel* model = infer_model_load(corrupted_path);
    assert(model == NULL);

    remove(corrupted_path);
}

static void test_infer_run_verification_samples_all_pass(void) {
    InferModel* model = infer_model_load("weights.bin");
    assert(model != NULL);

    FILE* f = fopen("verification_samples.bin", "rb");
    assert(f != NULL);

    char magic[4];
    assert(fread(magic, 1, 4, f) == 4);
    assert(memcmp(magic, "ICV1", 4) == 0);

    uint32_t n_samples, input_size, n_classes;
    assert(fread(&n_samples, sizeof(uint32_t), 1, f) == 1);
    assert(fread(&input_size, sizeof(uint32_t), 1, f) == 1);
    assert(fread(&n_classes, sizeof(uint32_t), 1, f) == 1);

    float* input = (float*)malloc(input_size * sizeof(float));
    float* expected_logits = (float*)malloc(n_classes * sizeof(float));
    float* expected_probs = (float*)malloc(n_classes * sizeof(float));
    assert(input != NULL && expected_logits != NULL && expected_probs != NULL);

    for (uint32_t s = 0; s < n_samples; s++) {
        assert(fread(input, sizeof(float), input_size, f) == input_size);
        assert(fread(expected_logits, sizeof(float), n_classes, f) == n_classes);
        assert(fread(expected_probs, sizeof(float), n_classes, f) == n_classes);
        uint32_t expected_class;
        assert(fread(&expected_class, sizeof(uint32_t), 1, f) == 1);

        InferResult result = infer_run(model, input, (int)input_size);

        int class_ok = result.predicted_class == (int)expected_class;
        float confidence_diff = fabsf(result.confidence - expected_probs[expected_class]);
        int confidence_ok = confidence_diff < 1e-4f;

        printf("sample %2u: predicted=%d expected=%u (%s) confidence=%.6f (%s)\n",
               (unsigned)s, result.predicted_class, (unsigned)expected_class,
               class_ok ? "ok" : "FAIL", (double)result.confidence,
               confidence_ok ? "ok" : "FAIL");

        assert(class_ok && confidence_ok);
    }

    free(input);
    free(expected_logits);
    free(expected_probs);
    fclose(f);
    infer_model_free(model);
}

static void test_infer_run_wrong_input_size_returns_sentinel(void) {
    InferModel* model = infer_model_load("weights.bin");
    assert(model != NULL);

    float input[783] = {0};
    InferResult result = infer_run(model, input, 783);

    assert(result.predicted_class == -1);
    assert(result.confidence == 0.0f);

    infer_model_free(model);
}

static void test_infer_run_null_model_returns_sentinel(void) {
    float input[784] = {0};
    InferResult result = infer_run(NULL, input, 784);

    assert(result.predicted_class == -1);
    assert(result.confidence == 0.0f);
}

static void test_infer_model_free_null_is_noop(void) {
    infer_model_free(NULL);
}

int main(void) {
    test_infer_model_load_valid_file();
    test_infer_model_load_missing_file_fails();
    test_infer_model_load_corrupted_checksum_fails();
    test_infer_run_verification_samples_all_pass();
    test_infer_run_wrong_input_size_returns_sentinel();
    test_infer_run_null_model_returns_sentinel();
    test_infer_model_free_null_is_noop();

    printf("All infer_c tests passed.\n");
    return 0;
}
