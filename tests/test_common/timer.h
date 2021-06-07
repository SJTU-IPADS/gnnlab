#ifndef TEST_COMMON_TIMER_H
#define TEST_COMMON_TIMER_H

#include <chrono>

class Timer {
 public:
  Timer(std::chrono::time_point<std::chrono::steady_clock> tp =
            std::chrono::steady_clock::now())
      : _start_time(tp) {}

  double Passed() {
    auto tp = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(tp -
                                                                  _start_time);
    return elapsed.count();
  }

  void Reset() { _start_time = std::chrono::steady_clock::now(); }

 private:
  std::chrono::time_point<std::chrono::steady_clock> _start_time;
};

#endif  // TEST_COMMON_TIMER_H
