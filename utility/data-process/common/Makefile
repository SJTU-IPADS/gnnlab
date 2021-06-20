CC=g++
FLAGS=-std=c++14 -fopenmp -fopt-info -Ofast -g -I
COMMON_FILES=loader/graph_loader.cc 
all: degrees

degrees: degrees/degrees.cc
	$(CC) $(FLAGS) degrees/degrees.cc -o degrees

create_alias_table: degrees.cc
	$(CC) $(FLAGS) degrees.cc -o degrees

.PHONY: clean
clean:
	rm -rf *.o degrees

.PHONY: FORCE
FORCE: