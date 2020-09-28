#include <gtest/gtest.h>

#include "estimation_utils.hpp"
#include "utils/optimization_problem.hpp"

template <tds::DiffMethod Method>
void test_ceres_estimation(double epsilon = 1e-4) {
  tds::CeresEstimator<Method, PendulumCost> estimator;
  estimator.parameters[0].minimum = 0.5;
  estimator.parameters[0].maximum = 10.;
  estimator.parameters[0].value = 3;
  estimator.parameters[1].minimum = 0.5;
  estimator.parameters[1].maximum = 10.;
  estimator.parameters[1].value = 5;
  estimator.options.max_num_iterations = 500;
  estimator.setup();
  auto summary = estimator.solve();
  std::cout << summary.FullReport() << std::endl;
  EXPECT_NEAR(estimator.parameters[0].value, true_link_lengths[0], epsilon);
  EXPECT_NEAR(estimator.parameters[1].value, true_link_lengths[1], epsilon);
}

TEST(CeresEstimation, FiniteDiff) {
  test_ceres_estimation<tds::DIFF_NUMERICAL>();
}

TEST(CeresEstimation, Ceres) { test_ceres_estimation<tds::DIFF_CERES>(); }

TEST(CeresEstimation, Dual) { test_ceres_estimation<tds::DIFF_DUAL>(0.1); }

TEST(CeresEstimation, StanReverse) {
  test_ceres_estimation<tds::DIFF_STAN_REVERSE>();
}

TEST(CeresEstimation, StanForward) {
  test_ceres_estimation<tds::DIFF_STAN_FORWARD>();
}