/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef SAMGRAPH_COMMON_TIMER_H
#define SAMGRAPH_COMMON_TIMER_H

#include <chrono>

namespace samgraph {
namespace common {

class Timer {
 public:
  Timer(std::chrono::time_point<std::chrono::steady_clock> tp =
            std::chrono::steady_clock::now())
      : _start_time(tp) {}

  template <typename T>
  bool Timeout(double count) const {
    return Passed<T>() >= count;
  }

  double Passed() const { return Passed<std::chrono::duration<double>>(); }

  double PassedSec() const { return Passed<std::chrono::seconds>(); }

  double PassedMicro() const { return Passed<std::chrono::microseconds>(); }

  double PassedNano() const { return Passed<std::chrono::nanoseconds>(); }

  template <typename T>
  double Passed() const {
    return Passed<T>(std::chrono::steady_clock::now());
  }

  template <typename T>
  double Passed(std::chrono::time_point<std::chrono::steady_clock> tp) const {
    const auto elapsed = std::chrono::duration_cast<T>(tp - _start_time);
    return elapsed.count();
  }

  uint64_t TimePointMicro() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
                   _start_time.time_since_epoch()).count();
  }

  void Reset() { _start_time = std::chrono::steady_clock::now(); }

 private:
  std::chrono::time_point<std::chrono::steady_clock> _start_time;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_COMMON_TIMER_H