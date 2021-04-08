#ifndef SAMGRAPH_COMMON_TIMER_H 
#define SAMGRAPH_COMMON_TIMER_H

#include <chrono>

namespace samgraph {
namespace common {

class Timer {
 public:
    Timer(std::chrono::time_point<std::chrono::steady_clock> tp = std::chrono::steady_clock::now())
        : _start_time(tp) {}

    template <typename T>
    bool Timeout(double count) const {
        return Passed<T>() >= count;
    }

    double Passed() const {
        return Passed<std::chrono::duration<double>>();
    }

    double PassedSec() const {
        return Passed<std::chrono::seconds>();
    }

    double PassedMicro() const {
        return Passed<std::chrono::microseconds>();
    }

    double PassedNano() const {
        return Passed<std::chrono::nanoseconds>();
    }

    template <typename T> double Passed() const {
        return Passed<T>(std::chrono::steady_clock::now());
    }

    template <typename T> double
    Passed(std::chrono::time_point<std::chrono::steady_clock> tp) const {
        const auto elapsed = std::chrono::duration_cast<T>(tp - _start_time);
        return elapsed.count();
    }

    void Reset() {
        _start_time = std::chrono::steady_clock::now();
    }

 private:
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_COMMON_TIMER_H