#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif
#include <vector>

int main() {
  std::vector<int> vec{3, 2, 4, 5, 1, 9, 0};
#ifdef __linux__
  __gnu_parallel::sort(vec.begin(), vec.end());
#else
  std::sort(vec.begin(), vec.end());
#endif

  for (int i = 0; i < vec.size(); i++) {
    printf("%d\n", vec[i]);
  }
}
