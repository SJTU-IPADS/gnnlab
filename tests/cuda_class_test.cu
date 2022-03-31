#include <cuda_runtime.h>
#include <gtest/gtest.h>

class A {
 public:
  A() : a(10) {}
  int a;

  __device__ void Print() { printf("%d\n", a); }
};

__global__ void accessA(A a) { a.Print(); }

TEST(CudaClassTest, ClassTest) {
  dim3 grid(1);
  dim3 block(1);

  A a;
  accessA<<<grid, block>>>(a);
}
