CC=/usr/local/cuda/bin/nvcc
FLAGS=-std=c++14

all: csrmm.out csrmm_bp.out compact_edge.out

csrmm.out: csrmm.cc common.h
	$(CC) $(FLAGS) -o csrmm.out -lcusparse csrmm.cc

csrmm_bp.out: csrmm_bp.cc common.h
	$(CC) $(FLAGS) -o csrmm_bp.out -lcusparse csrmm.cc

compact_edge.out: compact_edge.cu common.h
	$(CC) $(FLAGS) -o compact_edge.out -arch=sm_35 -lcusparse -I../3rdparty/cub -I. compact_edge.cu

.PHONY: clean
clean:
	rm -f *.o *.out