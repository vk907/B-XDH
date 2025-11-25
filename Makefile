# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native
LDFLAGS = -lcrypto

# Target executable
TARGET = bxdh

# Source files
SRC = test_bxdh.c

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
