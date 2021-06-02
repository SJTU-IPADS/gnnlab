BUILD_DIR := ./build
EXECUTABLE := samgraph_test
TARGET = $(BUILD_DIR)/$(EXECUTABLE)

all: test

test: $(TARGET)
	$(TARGET)

build: FORCE
	./build.sh

.PHONY: clean
clean:
	@rm -rf build
	@rm -rf chcore.out

.PHONY: FORCE
FORCE:
