#ifndef UTILITY_COMMON_LOADER_OPTIONS_H
#define UTILITY_COMMON_LOADER_OPTIONS_H

#include <argparse/argparse.hpp>
#include <string>
#include <unordered_map>

#include "graph_loader.h"

namespace utility {

class Options {
 public:
  Options(std::string app_name);
  void Parse(int argc, char *argv[]);

  std::string basic_path;
  GraphCode graph_code;

 private:
  argparse::ArgumentParser _arg_parser;
  static const std::unordered_map<std::string, GraphCode> _str2code;
};

}  // namespace utility

#endif  // UTILITY_COMMON_LOADER_OPTIONS_H
