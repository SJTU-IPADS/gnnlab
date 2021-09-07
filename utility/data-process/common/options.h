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
