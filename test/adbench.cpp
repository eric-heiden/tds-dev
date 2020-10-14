#include <benchmark/benchmark.h>

// clang-format off
// Differentiation must go first
#include "utils/differentiation.hpp"
// clang-format on
#include "math/neural_network.hpp"
#include "math/tiny/ceres_utils.h"
#include "math/tiny/cppad_utils.h"
#include "math/tiny/tiny_double_utils.h"
#include "utils/optimization_problem.hpp"

constexpr int kInputDim = 5;
const std::vector<int> kLayerSizes = {10, 10};
constexpr tds::NeuralNetworkActivation kActivation = tds::NN_ACT_ELU;
constexpr bool kLearnBias = true;
constexpr int kParameterDim = 175;

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

#define TDS_AD_BENCH(diff_type)                                   \
  static void BM_##diff_type##_Eval(benchmark::State& state) {    \
    using ProblemType =                                           \
        tds::OptimizationProblem<tds::diff_type, NNBenchFunctor>; \
    ProblemType problem;                                          \
    ProblemType::DoubleVector x(kParameterDim, 5.0);              \
    for (auto _ : state) {                                        \
      problem.fitness(x);                                         \
    }                                                             \
  }                                                               \
  BENCHMARK(BM_##diff_type##_Eval);                               \
                                                                  \
  static void BM_##diff_type##_Grad(benchmark::State& state) {    \
    using ProblemType =                                           \
        tds::OptimizationProblem<tds::diff_type, NNBenchFunctor>; \
    ProblemType problem;                                          \
    ProblemType::DoubleVector x(kParameterDim, 5.0);              \
    for (auto _ : state) {                                        \
      problem.gradient(x);                                        \
    }                                                             \
  }                                                               \
  BENCHMARK(BM_##diff_type##_Grad);

TDS_AD_BENCH(DIFF_NUMERICAL);
TDS_AD_BENCH(DIFF_CERES);
TDS_AD_BENCH(DIFF_STAN_REVERSE);
TDS_AD_BENCH(DIFF_STAN_FORWARD);
TDS_AD_BENCH(DIFF_CPPAD_AUTO);

// Run the benchmark
BENCHMARK_MAIN();
