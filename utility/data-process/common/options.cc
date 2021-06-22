#include "options.h"

#include <iostream>

namespace utility {

Options::Options(std::string app_name) : _app("", app_name) {
  root = "/graph-learning/samgraph/";
  graph = "papers100M";

  _app.add_option("-p,--root", root);
  _app.add_option("-g,--graph", graph)
      ->check(CLI::IsMember({
          "reddit",
          "products",
          "papers100M",
          "com-friendster",
      }));
}

void Options::Parse(int argc, char* argv[]) {
  _app.parse(argc, argv);

  std::cout << "Root: " << root << std::endl;
  std::cout << "Graph: " << graph << std::endl;
}

int Options::Exit(const CLI::ParseError& e) { return _app.exit(e); }

}  // namespace utility