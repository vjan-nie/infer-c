CC      = cc
CFLAGS  = -std=c99 -Wall -Wextra -Wpedantic -fPIC -Isrc
SRC_DIR = src
TEST_DIR = tests
BIN_DIR = bin
LIB_DIR = lib

all: test lib infer

$(BIN_DIR)/%.o: $(SRC_DIR)/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BIN_DIR)/test_matrix: $(BIN_DIR)/matrix.o $(TEST_DIR)/test_matrix.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(BIN_DIR)/matrix.o $(TEST_DIR)/test_matrix.c

$(BIN_DIR)/test_nn: $(BIN_DIR)/matrix.o $(BIN_DIR)/model.o $(BIN_DIR)/nn.o $(TEST_DIR)/test_nn.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(BIN_DIR)/matrix.o $(BIN_DIR)/model.o $(BIN_DIR)/nn.o $(TEST_DIR)/test_nn.c -lm

$(BIN_DIR)/test_infer_c: $(BIN_DIR)/matrix.o $(BIN_DIR)/model.o $(BIN_DIR)/nn.o $(BIN_DIR)/infer_c.o $(TEST_DIR)/test_infer_c.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(BIN_DIR)/matrix.o $(BIN_DIR)/model.o $(BIN_DIR)/nn.o $(BIN_DIR)/infer_c.o $(TEST_DIR)/test_infer_c.c -lm

$(BIN_DIR)/test_data: $(BIN_DIR)/data.o $(TEST_DIR)/test_data.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(BIN_DIR)/data.o $(TEST_DIR)/test_data.c -lm

$(BIN_DIR)/infer: $(BIN_DIR)/data.o $(BIN_DIR)/matrix.o $(BIN_DIR)/model.o \
                  $(BIN_DIR)/nn.o $(BIN_DIR)/infer_c.o $(SRC_DIR)/main.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BIN_DIR)/benchmark: $(BIN_DIR)/data.o $(BIN_DIR)/matrix.o $(BIN_DIR)/model.o \
                      $(BIN_DIR)/nn.o $(BIN_DIR)/infer_c.o $(SRC_DIR)/benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test: $(BIN_DIR)/test_matrix $(BIN_DIR)/test_nn $(BIN_DIR)/test_infer_c $(BIN_DIR)/test_data
	./$(BIN_DIR)/test_matrix
	./$(BIN_DIR)/test_nn
	./$(BIN_DIR)/test_infer_c
	./$(BIN_DIR)/test_data

valgrind: $(BIN_DIR)/test_matrix $(BIN_DIR)/test_nn $(BIN_DIR)/test_infer_c $(BIN_DIR)/test_data
	valgrind --leak-check=full --error-exitcode=1 ./$(BIN_DIR)/test_matrix
	valgrind --leak-check=full --error-exitcode=1 ./$(BIN_DIR)/test_nn
	valgrind --leak-check=full --error-exitcode=1 ./$(BIN_DIR)/test_infer_c
	valgrind --leak-check=full --error-exitcode=1 ./$(BIN_DIR)/test_data

benchmark: $(BIN_DIR)/benchmark
	./$(BIN_DIR)/benchmark data/raw/t10k-images-idx3-ubyte data/raw/t10k-labels-idx1-ubyte

infer: $(BIN_DIR)/infer

lib: $(LIB_DIR)/libinfer_c.a $(LIB_DIR)/libinfer_c.so

$(LIB_DIR)/libinfer_c.a: $(BIN_DIR)/matrix.o $(BIN_DIR)/model.o $(BIN_DIR)/nn.o $(BIN_DIR)/infer_c.o
	mkdir -p $(LIB_DIR)
	ar rcs $@ $^

$(LIB_DIR)/libinfer_c.so: $(BIN_DIR)/matrix.o $(BIN_DIR)/model.o $(BIN_DIR)/nn.o $(BIN_DIR)/infer_c.o
	mkdir -p $(LIB_DIR)
	$(CC) -shared -o $@ $^

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR) $(LIB_DIR)

.PHONY: all test valgrind clean lib benchmark infer
