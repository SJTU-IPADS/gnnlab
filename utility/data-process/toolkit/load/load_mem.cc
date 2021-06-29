#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

int main(int argc, char *argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph,
                                            utility::Options::is64type);
}