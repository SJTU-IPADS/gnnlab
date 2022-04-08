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