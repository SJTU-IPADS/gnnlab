CC=/usr/local/cuda/bin/nvcc
FLAGS=-std=c++14 -g -G

all: csrmm csrmm_bp compact_edge sample curand hash_table d2d macro timer inheritance sort

csrmm: csrmm.cc common.h
	$(CC) $(FLAGS) -o csrmm -lcusparse csrmm.cc

csrmm_bp: csrmm_bp.cc common.h
	$(CC) $(FLAGS) -o csrmm_bp -lcusparse csrmm.cc

compact_edge: compact_edge.cu common.h
	$(CC) $(FLAGS) -o compact_edge -arch=sm_35 -lcusparse -I../3rdparty/cub -I. compact_edge.cu

sample: sample.cu common.h
	$(CC) $(FLAGS) -o sample -arch=sm_35 -lcusparse -I../3rdparty/cub -I. sample.cu

curand:  curand.cu common.h
	$(CC) $(FLAGS) -o curand -arch=sm_35 -lcusparse -I../3rdparty/cub -I. curand.cu

hash_table:  hash_table.cu common.h hash_table.cuh
	$(CC) $(FLAGS) -o hash_table -arch=sm_35 -lcusparse -I../3rdparty/cub -I. hash_table.cu

d2d:  d2d.cu common.h
	$(CC) $(FLAGS) -o d2d -arch=sm_35 -lcusparse -I. d2d.cu

macro: macro.cc
	$(CC) -E -x c macro.cc -o macro

inheritance: inheritance.cc
	$(CC) $(FLAGS) inheritance.cc -o inheritance

timer: timer.cc
	$(CC) $(FLAGS) timer.cc -o timer

sort: sort.cc
	g++ -fopenmp sort.cc -o sort

.PHONY: clean
clean:
	rm -rf *.o *.so __pycache__