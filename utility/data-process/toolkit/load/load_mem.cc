#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

int main(int argc, char *argv[]) {
  utility::Options options("Graph property");
  OPTIONS_PARSE(options, argc, argv);

  utility::GraphLoader graph_loader(options.root);
  auto graph = graph_loader.GetGraphDataset(options.graph, options.is64type);
}