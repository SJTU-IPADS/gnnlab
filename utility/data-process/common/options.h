#ifndef UTILITY_COMMON_OPTIONS_H
#define UTILITY_COMMON_OPTIONS_H

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <string>
#include <unordered_map>

#include "graph_loader.h"

namespace utility {

#ifndef OPTIONS_PARSE
#define OPTIONS_PARSE(options, argc, argv) \
  try {                                    \
    (options).Parse((argc), (argv));       \
    (options).EnableOptions();             \
  } catch (const CLI::ParseError &e) {     \
    return (options).Exit(e);              \
  }
#endif

class Options {
 public:
  Options(std::string app_name);
  void Parse(int argc, char *argv[]);
  int Exit(const CLI::ParseError &e);
  void EnableOptions();

  std::string root;
  std::string graph;
  bool is64type;
  size_t num_threads;

 private:
  CLI::App _app;
};

}  // namespace utility

#endif  // UTILITY_COMMON_OPTIONS_H
