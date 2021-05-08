CC=g++
FLAGS=-std=c++14 -fopenmp -fopt-info -Ofast -g

all: com-friendster

com-friendster: com-friendster.cc
	$(CC) $(FLAGS) com-friendster.cc -o com-friendster 

.PHONY: clean
clean:
	rm -rf *.o com-friendster