CC=/usr/local/cuda/bin/nvcc
FLAGS=-std=c++14 -g -G

all: csrmm.out csrmm_bp.out compact_edge.out sample.out curand.out hash_table.out d2d.out

csrmm.out: csrmm.cc common.h
	$(CC) $(FLAGS) -o csrmm.out -lcusparse csrmm.cc

csrmm_bp.out: csrmm_bp.cc common.h
	$(CC) $(FLAGS) -o csrmm_bp.out -lcusparse csrmm.cc

compact_edge.out: compact_edge.cu common.h
	$(CC) $(FLAGS) -o compact_edge.out -arch=sm_35 -lcusparse -I../3rdparty/cub -I. compact_edge.cu

sample.out: sample.cu common.h
	$(CC) $(FLAGS) -o sample.out -arch=sm_35 -lcusparse -I../3rdparty/cub -I. sample.cu

curand.out:  curand.cu common.h
	$(CC) $(FLAGS) -o curand.out -arch=sm_35 -lcusparse -I../3rdparty/cub -I. curand.cu

hash_table.out:  hash_table.cu common.h hash_table.cuh
	$(CC) $(FLAGS) -o hash_table.out -arch=sm_35 -lcusparse -I../3rdparty/cub -I. hash_table.cu

d2d.out:  d2d.cu common.h
	$(CC) $(FLAGS) -o d2d.out -arch=sm_35 -lcusparse -I. d2d.cu

macro.out: macro.cc
	$(CC) -E -x c macro.cc -o macro.out

.PHONY: clean
clean:
	rm -rf *.o *.out *.so __pycache__