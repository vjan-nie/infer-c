CC      = cc
CFLAGS  = -std=c99 -Wall -Wextra -Wpedantic
SRC_DIR = src
TEST_DIR = tests
BIN_DIR = bin

# data.c/main.c are still empty placeholders and are intentionally not
# compiled (an empty translation unit warns under -Wpedantic).
all: test

$(BIN_DIR)/test_matrix: $(SRC_DIR)/matrix.c $(TEST_DIR)/test_matrix.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -o $@ $^

$(BIN_DIR)/test_nn: $(SRC_DIR)/matrix.c $(SRC_DIR)/model.c $(SRC_DIR)/nn.c $(TEST_DIR)/test_nn.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -o $@ $^ -lm

test: $(BIN_DIR)/test_matrix $(BIN_DIR)/test_nn
	./$(BIN_DIR)/test_matrix
	./$(BIN_DIR)/test_nn

valgrind: $(BIN_DIR)/test_matrix $(BIN_DIR)/test_nn
	valgrind --leak-check=full --error-exitcode=1 ./$(BIN_DIR)/test_matrix
	valgrind --leak-check=full --error-exitcode=1 ./$(BIN_DIR)/test_nn

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all test valgrind clean
