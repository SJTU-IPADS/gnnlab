#include "options.h"

namespace utility {

const std::unordered_map<std::string, GraphCode> Options::_str2code = {
    {"papers100M", kPapers100M},
    {"comfriendster", kComfriendster},
    {"reddit", kReddit},
    {"products", kProducts}};

Options::Options(std::string app_name) {
  _arg_parser = argparse::ArgumentParser(app_name);
  _arg_parser.add_argument("-c", "--graph_code")
      .default_value(std::string("0"))
      .required();

  _arg_parser.add_argument("-p", "--basic_path")
      .default_value(std::string())
      .required();
}

void Options::Parse(int argc, char* argv[]) {
  _arg_parser.parse_args(argc, argv);
  return;
}

}  // namespace utility