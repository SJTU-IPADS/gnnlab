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

#ifndef UTILITY_COMMON_OPTIONS_H
#define UTILITY_COMMON_OPTIONS_H

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <string>
#include <unordered_map>

#include "graph_loader.h"

namespace utility {

class Options {
 public:
  static void InitOptions(std::string app_name);
  template<typename T>
  static void CustomOption(std::string key, T& val) {
    _app.add_option(key, val);
  }
  static void Parse(int argc, char *argv[]);
  static int Exit(const CLI::ParseError &e);
  static void EnableOptions();

  static std::string root;
  static std::string graph;
  static bool is64type;
  static size_t num_threads;

 private:
  static CLI::App _app;
};

#ifndef OPTIONS_PARSE
#define OPTIONS_PARSE(argc, argv)            \
  try {                                      \
    utility::Options::Parse((argc), (argv)); \
    utility::Options::EnableOptions();       \
  } catch (const CLI::ParseError &e) {       \
    return utility::Options::Exit(e);        \
  }
#endif

}  // namespace utility

#endif  // UTILITY_COMMON_OPTIONS_H
