#include <gtest/gtest.h>

#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "utils/ceres_estimator.hpp"
#include "utils/pendulum.hpp"

const std::vector<double> true_link_lengths = {2, 4};

template <typename Algebra>
void rollout(const std::vector<typename Algebra::Scalar>& link_lengths,
             std::vector<std::vector<typename Algebra::Scalar>>* states) {
  using Scalar = typename Algebra::Scalar;
  tds::World<Algebra> world;
  tds::MultiBody<Algebra>* mb;
  Scalar dt = Algebra::fraction(1, 240);
  mb = world.create_multi_body();
  init_compound_pendulum<Algebra>(*mb, world, 2, link_lengths);
  typename Algebra::Vector3 gravity(Algebra::zero(), Algebra::zero(),
                                    Algebra::from_double(-9.81));
  for (int t = 0; t < 500; ++t) {
    tds::forward_dynamics(*mb, gravity);
    tds::integrate_euler(*mb, mb->q(), mb->qd(), mb->qdd(), dt);
    std::vector<Scalar> state(mb->dof());
    for (int i = 0; i < mb->dof(); ++i) {
      state[i] = mb->q(i);
    }
    states->push_back(state);
  }
}

template <typename Algebra>
class PendulumCost {
  using Scalar = typename Algebra::Scalar;

  std::vector<std::vector<double>> true_states_;

 public:
  static const int kDim = 2;

  PendulumCost() {
    rollout<tds::EigenAlgebra>(true_link_lengths, &true_states_);
  }

  Scalar operator()(const std::vector<Scalar>& link_lengths) const {
    std::vector<std::vector<Scalar>> states;
    rollout<Algebra>(link_lengths, &states);
    Scalar cost = Algebra::zero();
    for (std::size_t t = 0; t < states.size(); ++t) {
      for (int i = 0; i < kDim; ++i) {
        Scalar diff = states[t][i] - Algebra::from_double(true_states_[t][i]);
        cost += diff * diff;
      }
    }
    return cost;
  }
};

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