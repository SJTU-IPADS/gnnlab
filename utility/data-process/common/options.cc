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

#include "options.h"

#include <omp.h>

#include <iostream>

namespace utility {

std::string Options::root = "/graph-learning/samgraph/";
std::string Options::graph = "papers100M";
size_t Options::num_threads = 48;
bool Options::is64type = false;
CLI::App Options::_app;

void Options::InitOptions(std::string app_name) {
  _app.add_option("-p,--root", root);
  _app.add_option("-g,--graph", graph)
      ->check(CLI::IsMember({
          "reddit",
          "products",
          "papers100M",
          "com-friendster",
          "uk-2006-05",
          "twitter",
          "sk-2005",
      }));
  _app.add_option("-t,--threads", num_threads);
  _app.add_flag("--64", is64type);
}

void Options::Parse(int argc, char* argv[]) {
  _app.parse(argc, argv);

  std::cout << "Root: " << root << std::endl;
  std::cout << "Graph: " << graph << std::endl;
  std::cout << "Threads: " << num_threads << std::endl;
  std::cout << "64 bit: " << is64type << std::endl;
}

int Options::Exit(const CLI::ParseError& e) { return _app.exit(e); }

void Options::EnableOptions() { omp_set_num_threads(num_threads); }

}  // namespace utility