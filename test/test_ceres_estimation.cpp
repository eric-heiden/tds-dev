#include <gtest/gtest.h>

#include "estimation_utils.hpp"

template <tds::DiffMethod Method>
void test_ceres_estimation(double epsilon = 1e-4) {
  auto problem = create_problem<Method>();
  auto estimator = tds::CeresEstimator(&problem);
  estimator.setup();
  auto summary = estimator.solve();
  std::cout << summary.FullReport() << std::endl;
  EXPECT_NEAR(estimator.best_parameters()[0], true_link_lengths[0], epsilon);
  EXPECT_NEAR(estimator.best_parameters()[1], true_link_lengths[1], epsilon);
}

// clang-format off
TEST(CeresEstimation, FiniteDiff) {
  test_ceres_estimation<tds::DIFF_NUMERICAL>();
}

TEST(CeresEstimation, Ceres) {
  test_ceres_estimation<tds::DIFF_CERES>();
}

TEST(CeresEstimation, Dual) {
  test_ceres_estimation<tds::DIFF_DUAL>(0.1);
}

TEST(CeresEstimation, StanReverse) {
  test_ceres_estimation<tds::DIFF_STAN_REVERSE>();
}

TEST(CeresEstimation, StanForward) {
  test_ceres_estimation<tds::DIFF_STAN_FORWARD>();
}
// clang-format on
