#include "../src/matrix.h"
#include "../src/model.h"
#include "../src/nn.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_crc32_known_check_value(void) {
    const unsigned char input[] = "123456789";
    assert(model_crc32(input, 9) == 0xCBF43926U);
}

static void test_model_load_valid_file(void) {
    Model* model = model_load("weights.bin");
    assert(model != NULL);
    assert(model->n_layers == 2);
    assert(model->layers[0].weight.rows == 784);
    assert(model->layers[0].weight.cols == 128);
    assert(model->layers[1].weight.rows == 128);
    assert(model->layers[1].weight.cols == 10);
    model_free(model);
}

static void test_model_load_corrupted_checksum_fails(void) {
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

    /* Flip a bit well inside the first layer's weight data, away from
     * the trailing 4-byte checksum field, so the corruption is actually
     * caught by the checksum rather than coincidentally landing on an
     * unchecked byte. */
    buf[100] ^= 0xFFU;

    const char* corrupted_path = "bin/corrupted_weights.bin";
    FILE* dst = fopen(corrupted_path, "wb");
    assert(dst != NULL);
    assert(fwrite(buf, 1, (size_t)size, dst) == (size_t)size);
    fclose(dst);
    free(buf);

    Model* model = model_load(corrupted_path);
    assert(model == NULL);

    remove(corrupted_path);
}

static void append_u32(unsigned char* buf, size_t* offset, uint32_t v) {
    memcpy(buf + *offset, &v, sizeof(v));
    *offset += sizeof(v);
}

static void append_f32(unsigned char* buf, size_t* offset, float v) {
    memcpy(buf + *offset, &v, sizeof(v));
    *offset += sizeof(v);
}

static void test_model_load_chain_incompatible_fails(void) {
    /* Layer 0: file rows=4,cols=3 (out=4,in=3). Layer 1: file rows=2,cols=5
     * (out=2,in=5). After model_load's transpose, layers[0].weight.cols
     * == 4 but layers[1].weight.rows == 5: the chain doesn't match, even
     * though the checksum below is computed correctly over this content. */
    const uint32_t l0_rows = 4, l0_cols = 3;
    const uint32_t l1_rows = 2, l1_cols = 5;
    size_t content_len = 12 /* header */
                        + 12 + (size_t)l0_rows * l0_cols * 4 + (size_t)l0_rows * 4
                        + 12 + (size_t)l1_rows * l1_cols * 4 + (size_t)l1_rows * 4;
    size_t total_len = content_len + 4;

    unsigned char* buf = (unsigned char*)malloc(total_len);
    assert(buf != NULL);
    size_t offset = 0;
    memcpy(buf + offset, "IC01", 4);
    offset += 4;
    append_u32(buf, &offset, 1); /* format_version */
    append_u32(buf, &offset, 2); /* n_layers */

    append_u32(buf, &offset, 0); /* activation_id */
    append_u32(buf, &offset, l0_rows);
    append_u32(buf, &offset, l0_cols);
    for (uint32_t k = 0; k < l0_rows * l0_cols; k++) {
        append_f32(buf, &offset, 0.0f);
    }
    for (uint32_t k = 0; k < l0_rows; k++) {
        append_f32(buf, &offset, 0.0f);
    }

    append_u32(buf, &offset, 0); /* activation_id */
    append_u32(buf, &offset, l1_rows);
    append_u32(buf, &offset, l1_cols);
    for (uint32_t k = 0; k < l1_rows * l1_cols; k++) {
        append_f32(buf, &offset, 0.0f);
    }
    for (uint32_t k = 0; k < l1_rows; k++) {
        append_f32(buf, &offset, 0.0f);
    }

    assert(offset == content_len);
    uint32_t crc = model_crc32(buf, content_len);
    append_u32(buf, &offset, crc);
    assert(offset == total_len);

    const char* path = "bin/chain_incompatible_weights.bin";
    FILE* dst = fopen(path, "wb");
    assert(dst != NULL);
    assert(fwrite(buf, 1, total_len, dst) == total_len);
    fclose(dst);
    free(buf);

    Model* model = model_load(path);
    assert(model == NULL);

    remove(path);
}

static void test_model_load_oversized_dims_fails(void) {
    /* A single layer header claiming rows=2^31 (> INT_MAX). No payload
     * bytes are needed: the bound check rejects this before any payload
     * read is attempted. */
    size_t content_len = 12 /* header */ + 12 /* layer header */;
    size_t total_len = content_len + 4;

    unsigned char* buf = (unsigned char*)malloc(total_len);
    assert(buf != NULL);
    size_t offset = 0;
    memcpy(buf + offset, "IC01", 4);
    offset += 4;
    append_u32(buf, &offset, 1); /* format_version */
    append_u32(buf, &offset, 1); /* n_layers */

    append_u32(buf, &offset, 0);          /* activation_id */
    append_u32(buf, &offset, 0x80000000U); /* rows: 2^31 > INT_MAX */
    append_u32(buf, &offset, 1);          /* cols */

    assert(offset == content_len);
    uint32_t crc = model_crc32(buf, content_len);
    append_u32(buf, &offset, crc);
    assert(offset == total_len);

    const char* path = "bin/oversized_dims_weights.bin";
    FILE* dst = fopen(path, "wb");
    assert(dst != NULL);
    assert(fwrite(buf, 1, total_len, dst) == total_len);
    fclose(dst);
    free(buf);

    Model* model = model_load(path);
    assert(model == NULL);

    remove(path);
}

static void test_chain_compatible_valid(void) {
    Layer layers[2];
    layers[0].weight = mat_create(3, 4);
    layers[0].bias = mat_create(1, 4);
    layers[0].activation_id = 1;
    layers[1].weight = mat_create(4, 2);
    layers[1].bias = mat_create(1, 2);
    layers[1].activation_id = 2;

    Model model;
    model.layers = layers;
    model.n_layers = 2;

    assert(model_layers_chain_compatible(&model) == 1);

    mat_free(&layers[0].weight);
    mat_free(&layers[0].bias);
    mat_free(&layers[1].weight);
    mat_free(&layers[1].bias);
}

static void test_chain_compatible_invalid(void) {
    Layer layers[2];
    layers[0].weight = mat_create(3, 4);
    layers[0].bias = mat_create(1, 4);
    layers[0].activation_id = 1;
    layers[1].weight = mat_create(5, 2);
    layers[1].bias = mat_create(1, 2);
    layers[1].activation_id = 2;

    Model model;
    model.layers = layers;
    model.n_layers = 2;

    assert(model_layers_chain_compatible(&model) == 0);

    mat_free(&layers[0].weight);
    mat_free(&layers[0].bias);
    mat_free(&layers[1].weight);
    mat_free(&layers[1].bias);
}

static void test_relu_mixed_inputs(void) {
    Matrix m = mat_create(1, 4);
    float vals[4] = {-2.0f, 0.0f, 3.0f, -0.5f};
    for (int i = 0; i < 4; i++) {
        m.data[i] = vals[i];
    }

    nn_relu(&m);

    assert(m.data[0] == 0.0f);
    assert(m.data[1] == 0.0f);
    assert(m.data[2] == 3.0f);
    assert(m.data[3] == 0.0f);

    mat_free(&m);
}

static void test_softmax_known_values(void) {
    Matrix in = mat_create(1, 3);
    Matrix out = mat_create(1, 3);
    in.data[0] = 0.0f;
    in.data[1] = 0.0f;
    in.data[2] = 0.0f;

    nn_softmax(&in, &out);

    for (int i = 0; i < 3; i++) {
        assert(fabsf(out.data[i] - (1.0f / 3.0f)) < 1e-6f);
    }

    mat_free(&in);
    mat_free(&out);
}

static void test_softmax_large_magnitude_stable(void) {
    Matrix in = mat_create(1, 3);
    Matrix out = mat_create(1, 3);
    in.data[0] = 1000.0f;
    in.data[1] = 1.0f;
    in.data[2] = 0.0f;

    nn_softmax(&in, &out);

    float sum = out.data[0] + out.data[1] + out.data[2];
    assert(isfinite(out.data[0]) && isfinite(out.data[1]) && isfinite(out.data[2]));
    assert(fabsf(sum - 1.0f) < 1e-4f);
    assert(out.data[0] > out.data[1]);
    assert(out.data[0] > out.data[2]);

    mat_free(&in);
    mat_free(&out);
}

static void test_verification_samples_all_pass(void) {
    Model* model = model_load("weights.bin");
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

    Matrix input = mat_create(1, (int)input_size);
    ForwardResult result;
    result.logits = mat_create(1, (int)n_classes);
    result.probs = mat_create(1, (int)n_classes);

    float* expected_logits = (float*)malloc(n_classes * sizeof(float));
    float* expected_probs = (float*)malloc(n_classes * sizeof(float));
    assert(expected_logits != NULL && expected_probs != NULL);

    for (uint32_t s = 0; s < n_samples; s++) {
        assert(fread(input.data, sizeof(float), input_size, f) == input_size);
        assert(fread(expected_logits, sizeof(float), n_classes, f) == n_classes);
        assert(fread(expected_probs, sizeof(float), n_classes, f) == n_classes);
        uint32_t expected_class;
        assert(fread(&expected_class, sizeof(uint32_t), 1, f) == 1);

        nn_forward(model, &input, &result);

        float max_logit_diff = 0.0f;
        float max_prob_diff = 0.0f;
        for (uint32_t j = 0; j < n_classes; j++) {
            float ld = fabsf(result.logits.data[j] - expected_logits[j]);
            float pd = fabsf(result.probs.data[j] - expected_probs[j]);
            if (ld > max_logit_diff) {
                max_logit_diff = ld;
            }
            if (pd > max_prob_diff) {
                max_prob_diff = pd;
            }
        }

        int predicted_class = 0;
        float best = result.logits.data[0];
        for (uint32_t j = 1; j < n_classes; j++) {
            if (result.logits.data[j] > best) {
                best = result.logits.data[j];
                predicted_class = (int)j;
            }
        }

        int logits_ok = max_logit_diff < 1e-3f;
        int probs_ok = max_prob_diff < 1e-4f;
        int class_ok = predicted_class == (int)expected_class;

        printf("sample %2u: logits_max_diff=%.6f (%s) probs_max_diff=%.6f (%s) "
               "predicted=%d expected=%u (%s)\n",
               (unsigned)s, (double)max_logit_diff, logits_ok ? "ok" : "FAIL",
               (double)max_prob_diff, probs_ok ? "ok" : "FAIL",
               predicted_class, (unsigned)expected_class, class_ok ? "ok" : "FAIL");

        assert(logits_ok && probs_ok && class_ok);
    }

    free(expected_logits);
    free(expected_probs);
    mat_free(&input);
    mat_free(&result.logits);
    mat_free(&result.probs);
    fclose(f);
    model_free(model);
}

int main(void) {
    test_crc32_known_check_value();
    test_model_load_valid_file();
    test_model_load_corrupted_checksum_fails();
    test_model_load_chain_incompatible_fails();
    test_model_load_oversized_dims_fails();
    test_chain_compatible_valid();
    test_chain_compatible_invalid();
    test_relu_mixed_inputs();
    test_softmax_known_values();
    test_softmax_large_magnitude_stable();
    test_verification_samples_all_pass();

    printf("All nn tests passed.\n");
    return 0;
}
