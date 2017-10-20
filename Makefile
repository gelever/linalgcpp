CXX := g++
CXX_FLAGS := -g -O1 -Wall -Werror
LIBS := -llapack -lblas -lm

SRC_DIR := src
OBJ_DIR := obj
BUILD_DIR := build
TEST_DIR := tests

SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(SRC)))
PROGS := 
TESTS := test

all: $(OBJS) $(PROGS)

$(PROGS): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) $^ $@.cpp $(LIBS) -o $(BUILD_DIR)/$@

$(TESTS): $(OBJS)
	@mkdir -p $(BUILD_DIR)/tests
	$(CXX) $(CXX_FLAGS) $^ $(TEST_DIR)/$@.cpp $(LIBS) -o $(BUILD_DIR)/tests/$@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(C_FLAGS) -c $< $(LIBS) -o $@

clean:
	@rm -rf $(OBJ_DIR)/*.o
	@rm -rf $(addprefix $(BUILD_DIR)/, $(PROGS))
	@rm -rf $(addprefix $(BUILD_DIR)/tests, $(tests))

phony: all clean
