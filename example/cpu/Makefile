CC=g++
FLAGS=-std=c++14 -g -I../../3rdparty/parallel-hashmap
INCLUDES = $(wildcard ../../samgraph-cpu/*.h)

all: sample.out

sample.out: sample.cc $(INCLUDES)
	$(CC) $(FLAGS) -o sample.out sample.cc

.PHONY: clean
clean:
	rm -f *.o *.out