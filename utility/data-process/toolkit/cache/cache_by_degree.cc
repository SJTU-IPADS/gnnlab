#include <fstream>
#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif

#include "common/graph_loader.h"
#include "common/options.h"
#include "common/utils.h"

void randkingNodesToFile(utility::GraphPtr graph,
                         std::shared_ptr<utility::DegreeInfo> info) {
  size_t num_nodes = graph->num_nodes;
  const std::vector<uint32_t>& out_degrees = info->out_degrees;
  std::vector<std::pair<uint32_t, uint32_t>> outdegree_id_list(num_nodes);

#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    outdegree_id_list[i] = {out_degrees[i], i};
  }

#ifdef __linux__
  __gnu_parallel::sort(outdegree_id_list.begin(), outdegree_id_list.end(),
                       std::greater<std::pair<uint32_t, uint32_t>>());
#else
  std::sort(outdegree_id_list.begin(), outdegree_id_list.end(),
            std::greater<std::pair<uint32_t, uint32_t>>());
#endif

  std::vector<uint32_t> ranking_nodes(num_nodes);

#pragma omp parallel for
  for (size_t i = 0; i < num_nodes; i++) {
    ranking_nodes[i] = outdegree_id_list[i].second;
  }

  std::ofstream ofs(
      graph->folder + "cache_by_degree.bin",
      std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

  ofs.write((const char*)ranking_nodes.data(),
            ranking_nodes.size() * sizeof(uint32_t));

  ofs.close();
}

int main(int argc, char* argv[]) {
  utility::Options::InitOptions("Graph property");
  OPTIONS_PARSE(argc, argv);

  utility::GraphLoader graph_loader(utility::Options::root);
  auto graph = graph_loader.GetGraphDataset(utility::Options::graph);
  auto degree_info = utility::DegreeInfo::GetDegrees(graph);

  randkingNodesToFile(graph, degree_info);
}