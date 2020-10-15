// clang-format off
// Differentiation must go first
#include "utils/differentiation.hpp"
// clang-format on
#include "math/neural_network.hpp"
#include "math/tiny/ceres_utils.h"
#include "math/tiny/cppad_utils.h"
#include "math/tiny/tiny_double_utils.h"
#include "utils/optimization_problem.hpp"
#include "estimation_utils.hpp"
#include <random>

constexpr int kInputDim = 5;
const std::vector<int> kLayerSizes = {10, 10};
constexpr tds::NeuralNetworkActivation kActivation = tds::NN_ACT_TANH;
constexpr bool kLearnBias = true;
constexpr int kParameterDim = 175;
constexpr int kTrials = 10;

template <typename Algebra>
std::vector<typename Algebra::Scalar> Input() {
  static std::vector<typename Algebra::Scalar> input(kInputDim,
                                                     Algebra::from_double(5.0));
  return input;
}

template <typename Algebra>
struct NNBenchFunctor {
  using Scalar = typename Algebra::Scalar;
  static constexpr int kDim = kParameterDim;

  mutable tds::NeuralNetwork<Algebra> net_;
  mutable std::vector<Scalar> output_;
  NNBenchFunctor() : net_(kInputDim, kLayerSizes, kActivation, kLearnBias) {
    if (net_.num_parameters() != kParameterDim) {
      std::cout << "kParameterDim should be " << net_.num_parameters() << "\n";
      std::exit(1);
    }
  }
  Scalar operator()(const std::vector<Scalar>& x) const {
    net_.set_parameters(x);
    net_.compute(Input<Algebra>(), output_);
    Scalar value = Algebra::zero();
    for (auto& output : output_) {
      value += output;
    }
    return value;
  }
};

#define TDS_AD_BENCH(diff_type)                                         \
  static void BM_##diff_type##_NN_Grad(benchmark::State& state) {       \
    using ProblemType =                                                 \
        tds::OptimizationProblem<tds::diff_type, NNBenchFunctor>;       \
    ProblemType problem;                                                \
    ProblemType::DoubleVector x(kParameterDim, 5.0);                    \
    problem.gradient(x);                                                \
    for (auto _ : state) {                                              \
      problem.gradient(x);                                              \
    }                                                                   \
  }                                                                     \
  BENCHMARK(BM_##diff_type##_NN_Grad);                                  \
                                                                        \
  static void BM_##diff_type##_Pendulum_Grad(benchmark::State& state) { \
    auto problem = create_problem<tds::diff_type>();                    \
    using ProblemType = decltype(problem);                              \
    ProblemType::DoubleVector x(ProblemType::kParameterDim, 5.0);       \
    problem.gradient(x);                                                \
    for (auto _ : state) {                                              \
      problem.gradient(x);                                              \
    }                                                                   \
  }                                                                     \
  BENCHMARK(BM_##diff_type##_Pendulum_Grad);

#define TDS_AD_TEST(diff_type)                                               \
  TEST(ADTest, diff_type##_VS_CERES) {                                       \
    std::default_random_engine e(1234);                                      \
    std::normal_distribution<> normal_dist(0, 1);                            \
    using DiffProblemType =                                                  \
        tds::OptimizationProblem<tds::diff_type, NNBenchFunctor>;            \
    using CeresProblemType =                                                 \
        tds::OptimizationProblem<tds::DIFF_CERES, NNBenchFunctor>;           \
    DiffProblemType dproblem;                                                \
    CeresProblemType cproblem;                                               \
    DiffProblemType::DoubleVector dx(kParameterDim);                         \
    CeresProblemType::DoubleVector cx(kParameterDim);                        \
    for (int trial = 0; trial < kTrials; ++trial) {                          \
      for (int i = 0; i < kParameterDim; ++i) {                              \
        const double xi = normal_dist(e);                                    \
        dx[i] = xi;                                                          \
        cx[i] = xi;                                                          \
      }                                                                      \
      auto dgrad = dproblem.gradient(dx);                                    \
      auto cgrad = cproblem.gradient(cx);                                    \
      EXPECT_THAT(dgrad, ::testing::Pointwise(::testing::FloatEq(), cgrad)); \
    }                                                                        \
  }
