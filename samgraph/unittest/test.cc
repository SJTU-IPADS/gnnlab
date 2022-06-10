#include "gtest/gtest.h"
#include "common/run_config.h"

TEST(gTest, equal) {
  EXPECT_EQ(1, 1);
}

TEST(gTest, not_equal) {
  EXPECT_EQ(1, 1);
}

TEST(TestRunConfig, is_configured) {
  EXPECT_EQ(samgraph::common::RunConfig::is_configured, false);
}
