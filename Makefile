CXX := g++
CXX_FLAGS := -g -O1 -Wall -Werror
LIBS := -llapack -lblas -lm

SRC_DIR := src
OBJ_DIR := obj
BUILD_DIR := build

SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(SRC)))
PROGS := test

all: $(OBJS) $(PROGS)

$(PROGS): $(OBJS)
	@echo $@
	$(CXX) $(CXX_FLAGS) $^ $@.cpp $(LIBS) -o $(BUILD_DIR)/$@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(C_FLAGS) -c $< $(LIBS) -o $@

clean:
	@rm -rf $(OBJ_DIR)/*.o
	@rm -rf $(addprefix $(BUILD_DIR)/, $(PROGS))

phony: all clean
