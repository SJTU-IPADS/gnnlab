#include "common/operation.h"
#include "common/common.h"
#include "common/engine.h"
#include "common/constant.h"
#include "common/logging.h"
#include <CLI/CLI.hpp>
CLI::App _app;
namespace {
using namespace samgraph;
using namespace common;
using samgraph::common::IdType;
std::unordered_map<std::string, samgraph::common::SampleType> sample_type_strs = {
  {"khop0",                    samgraph::common::kKHop0},
  {"khop1",                    samgraph::common::kKHop1},
  {"weighted_khop",            samgraph::common::kWeightedKHop},
  {"random_walk",              samgraph::common::kRandomWalk},
  {"weighted_khop_prefix",     samgraph::common::kWeightedKHopPrefix},
  {"khop2",                    samgraph::common::kKHop2},
  {"weighted_khop_hash_dedup", samgraph::common::kWeightedKHopHashDedup},
};

std::unordered_map<std::string, samgraph::common::CachePolicy> cache_policy_strs = {
  {"degree"          ,  samgraph::common::kCacheByDegree},
  {"heuristic"       ,  samgraph::common::kCacheByHeuristic},
  {"pre_sample"      ,  samgraph::common::kCacheByPreSample},
  {"degree_hop"      ,  samgraph::common::kCacheByDegreeHop},
  {"presample_static",  samgraph::common::kCacheByPreSampleStatic},
  {"fake_optimal"    ,  samgraph::common::kCacheByFakeOptimal},
  {"dynamic_cache"   ,  samgraph::common::kDynamicCache},
  {"random"          ,  samgraph::common::kCacheByRandom},
};

std::unordered_map<std::string, std::string> configs;
std::vector<std::string> fanout_vec = {"25", "10"};
std::string env_profile_level = "3";
std::string env_log_level = "warn";
std::string env_empty_feat = "0";

void InitOptions(std::string app_name) {
  configs = {
    {"_arch",  "3"},
    {"arch",  "arch3"},
    {"_cache_policy",  std::to_string((int)samgraph::common::kCacheByPreSample)},
    {"cache_policy",  "pre_sample"},
    {"_sample_type",  std::to_string((int)samgraph::common::SampleType::kKHop2)},
    {"sample_type",  "khop2"},
    {"barriered_epoch",  "1"},
    {"batch_size",  "8000"},
    {"cache_percentage",  "0"},
    {"dataset_path",  "/graph-learning/samgraph/products"},
    {"root_path",  "/graph-learning/samgraph/"},
    {"dataset",  "products"},
    {"fanout",  "25 10"},
    {"have_switcher",  "0"},
    {"max_copying_jobs",  "1"},
    {"max_sampling_jobs",  "1"},
    {"num_epoch",  "3"},
    {"num_fanout",  "2"},
    {"num_layer",  "2"},
    {"num_sample_worker",  "1"},
    {"num_train_worker",  "1"},
    {"num_worker",  "1"},
    {"omp_thread_num",  "40"},
    {"presample_epoch",  "1"},
    {"random_walk_length",  "3"},
    {"random_walk_restart_prob",  "0.5"},
    {"num_random_walk",  "4"},
    {"num_neighbor",  "5"},
    {"trainer_ctx",  "cuda:0"},
    {"sampler_ctx",  "cuda:1"},
    {"worker_id",  "0"},
  };
  _app.add_option("--arch", configs["arch"])
      ->check(CLI::IsMember({
          "arch0",
          "arch1",
          "arch2",
          "arch3",
      }));
  _app.add_option("--sample-type", configs["sample_type"])
      ->check(CLI::IsMember({
          "khop0",
          "khop1",
          "weighted_khop",
          "random_walk",
          "weighted_khop_prefix",
          "khop2",
          "weighted_khop_hash_dedup",
      }));
  _app.add_option("--max-sampling-jobs", configs["max_sampling_jobs"]);
  _app.add_option("--max-copying-jobs", configs["max_copying_jobs"]);
  _app.add_option("--cache-policy", configs["cache_policy"])
      ->check(CLI::IsMember({
          "degree",
          "heuristic",
          "pre_sample",
          "degree_hop",
          "presample_static",
          "fake_optimal",
          "dynamic_cache",
          "random",
      }));
  _app.add_option("--cache-percentage", configs["cache_percentage"]);
  _app.add_option("--num-epoch", configs["num_epoch"]);
  _app.add_option("--batch-size", configs["batch_size"]);
  _app.add_option("--barriered-epoch",            configs["barriered_epoch"]);
  _app.add_option("--root-path",                  configs["root_path"]);
  _app.add_option("--dataset",                    configs["dataset"]);
  _app.add_option("--fanout",                     fanout_vec);
  _app.add_option("--num-sample-worker",          configs["num_sample_worker"]);
  _app.add_option("--num-train-worker",           configs["num_train_worker"]);
  _app.add_option("--num-worker",                 configs["num_worker"]);
  _app.add_option("--omp-thread-num",             configs["omp_thread_num"]);
  _app.add_option("--presample-epoch",            configs["presample_epoch"]);
  _app.add_option("--random-walk-length",         configs["random_walk_length"]);
  _app.add_option("--random-walk-restart-prob",   configs["random_walk_restart_prob"]);
  _app.add_option("--num-random-walk",            configs["num_random_walk"]);
  _app.add_option("--num-neighbor",               configs["num_neighbor"]);
  _app.add_option("--trainer-ctx",                configs["trainer_ctx"]);
  _app.add_option("--sampler-ctx",                configs["sampler_ctx"]);
  _app.add_option("--profile-level",              env_profile_level);
  _app.add_option("--log-level",                  env_log_level);
  _app.add_option("--empty-feat",                 env_empty_feat);
  _app.add_flag_callback("--pipeline",    [](){configs["pipeline"]="True";});
  _app.add_flag_callback("--no-pipeline", [](){configs["pipeline"]="False";});
}
void Parse(int argc, char** argv) {
  try {
    _app.parse(argc, argv);
  } catch(const CLI::ParseError &e) {
    _app.exit(e);
    exit(1);
  }
  if (configs["arch"] == "arch0") {
    configs["sampler_ctx"] = "cpu:0";
  } else if (configs["arch"] == "arch1" || configs["arch"] == "arch2") {
    configs["sampler_ctx"] = "cuda:0";
  } else {
    configs["sampler_ctx"] = "cuda:1";
  }
  configs["_arch"] = configs["arch"].substr(4, 1);
  configs["_sample_type"] = std::to_string((int)sample_type_strs[configs["sample_type"]]);
  configs["_cache_policy"] = std::to_string((int)cache_policy_strs[configs["cache_policy"]]);
  configs["fanout"] = CLI::detail::join(fanout_vec, " ");
  configs["num_fanout"] = std::to_string(fanout_vec.size());
  configs["num_layer"] = std::to_string(fanout_vec.size());
  configs["dataset_path"] = configs["root_path"] + configs["dataset"];
  setenv(samgraph::common::Constant::kEnvProfileLevel.c_str(), env_profile_level.c_str(), 1);
  setenv("SAMGRAPH_LOG_LEVEL", env_log_level.c_str(), 1);
  setenv(samgraph::common::Constant::kEnvEmptyFeat.c_str(), env_empty_feat.c_str(), 1);

  std::cout << "('arch', "              << configs["arch"]              << ")\n";
  std::cout << "('pipeline', "          << configs["pipeline"]       << ")\n";
  std::cout << "('sample_type', "       << configs["sample_type"]       << ")\n";
  std::cout << "('dataset_path', '"     << configs["dataset_path"]      << "')\n";
  std::cout << "('cache_policy', "      << configs["cache_policy"]      << ")\n";
  std::cout << "('cache_percentage', "  << configs["cache_percentage"]  << ")\n";
  std::cout << "('max_sampling_jobs', " << configs["max_sampling_jobs"] << ")\n";
  std::cout << "('max_copying_jobs', "  << configs["max_copying_jobs"]  << ")\n";
  std::cout << "('num_epoch', "         << configs["num_epoch"]         << ")\n";
  std::cout << "('fanout', ["           << CLI::detail::join(fanout_vec, ",") << "])\n";
  std::cout << "('batch_size', "        << configs["batch_size"]        << ")\n";
  std::cout << "('num_hidden', "        << 0                            << ")\n";
  std::cout << "('lr', "                << 0                            << ")\n";
  std::cout << "('dropout', "           << 0                            << ")\n";
  std::cout << "('weight_decay', "      << 0                            << ")\n";
  std::cout << "('sampler_gpu', "       << "'TBD'"     << ")\n";
  std::cout << "('trainer_gpu', "       << "'TBD'"     << ")\n";

  configs["num_epoch"] = std::to_string(std::stoi(configs["num_epoch"])+1);
}

};

int main(int argc, char** argv) {
  InitOptions("");
  Parse(argc, argv);
  size_t num_epoch = std::stoi(configs["num_epoch"]);
  samgraph::common::samgraph_config_from_map(configs);
  samgraph::common::samgraph_init();
  for (size_t i = 0; i < num_epoch; i++) {
    for (size_t b = 0; b < samgraph::common::samgraph_steps_per_epoch(); b++) {
      samgraph::common::samgraph_sample_once();
      samgraph::common::samgraph_get_next_batch();
      // samgraph::common::samgraph_report_step(i, b);
    }
    // samgraph::common::samgraph_report_epoch(i);
  }
  samgraph::common::samgraph_report_step_average(num_epoch-1, samgraph::common::samgraph_steps_per_epoch()-1);
  samgraph::common::samgraph_report_epoch_average(num_epoch-1);
  samgraph::common::samgraph_report_init();
  samgraph::common::samgraph_report_node_access();
  samgraph::common::samgraph_dump_trace();
  samgraph::common::samgraph_shutdown();
}
