#include "options.h"

#include <omp.h>

#include <iostream>

namespace utility {

Options::Options(std::string app_name) : _app("", app_name) {
  root = "/graph-learning/samgraph/";
  graph = "papers100M";
  num_threads = 48;
  is64type = false;

  _app.add_option("-p,--root", root);
  _app.add_option("-g,--graph", graph)
      ->check(CLI::IsMember({
          "reddit",
          "products",
          "papers100M",
          "com-friendster",
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