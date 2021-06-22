#ifndef UTILITY_COMMON_UTILS_H
#define UTILITY_COMMON_UTILS_H

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace utility {

inline bool FileExist(const std::string &filepath) {
  std::ifstream f(filepath);
  return f.good();
}

inline void Check(bool cond, std::string error_msg = "") {
  if (!cond) {
    std::cout << error_msg << std::endl;
    exit(1);
  }
}

class Timer {
 public:
  Timer(std::chrono::time_point<std::chrono::steady_clock> tp =
            std::chrono::steady_clock::now())
      : _start_time(tp) {}

  double Passed() const {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - _start_time);
    return elapsed.count();
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> _start_time;
};

}  // namespace utility

#endif  // UTILITY_COMMON_UTILS_H