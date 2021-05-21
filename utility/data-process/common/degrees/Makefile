CC=g++
FLAGS=-std=c++14 -fopenmp -fopt-info -Ofast -g

all: degrees

degrees: degrees.cc
	$(CC) $(FLAGS) degrees.cc -o degrees

.PHONY: clean
clean:
	rm -rf *.o degrees