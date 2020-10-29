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
#include "world.hpp"
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "urdf/system_constructor.hpp"
#include "urdf/urdf_cache.hpp"
#include "utils/file_utils.hpp"
#include "mb_constraint_solver_spring.hpp"
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

template <typename Algebra, bool UseSpringContact>
struct ContactModelFunctor {
  static const inline int kDim = 5;
  using Scalar = typename Algebra::Scalar;

  std::string urdf_filename;
  std::string plane_filename;

  ContactModelFunctor() {
    tds::FileUtils::find_file("pendulum5.urdf", urdf_filename);
    tds::FileUtils::find_file("plane_implicit.urdf", plane_filename);
  }

  Scalar operator()(const std::vector<Scalar>& x) const {
    tds::World<Algebra> world;
    if constexpr (UseSpringContact) {
      world.set_mb_constraint_solver(
          new tds::MultiBodyConstraintSolverSpring<Algebra>);
    }
    tds::UrdfCache<Algebra> cache;
    auto* system = cache.construct(urdf_filename, world, false, false);
    system->base_X_world().translation = Algebra::unit3_z();
    for (int i = 0; i < system->dof() && i < static_cast<int>(x.size()); ++i) {
      system->qd(i) = x[i];
    }
    Scalar mse = Algebra::zero();
    Scalar dt = Algebra::fraction(1, 1000);
    int step_limit = 5000;
    for (int t = 0; t < step_limit; ++t) {
      tds::forward_dynamics(*system, world.get_gravity());
      world.step(dt);
      tds::integrate_euler(*system, dt);
      if constexpr (std::is_same_v<Scalar, double>) {
        if (t % 100 == 0) {
          system->print_state();
        }
      }
      for (int i = 0; i < system->dof(); ++i) {
        mse += system->q(i) / Algebra::from_double(system->dof() * step_limit);
      }
    }
    return mse;
  }
};

template <typename Algebra>
using LCPContactModelFunctor = ContactModelFunctor<Algebra, false>;
template <typename Algebra>
using SpringContactModelFunctor = ContactModelFunctor<Algebra, true>;


#define TDS_AD_BENCH(diff_type)                                         \
                                                                        \
  static void BM_##diff_type##_NN_Grad(benchmark::State& state) {       \
    using ProblemType =                                                 \
        tds::OptimizationProblem<tds::diff_type, NNBenchFunctor>;       \
    if constexpr (tds::diff_type == tds::DIFF_CPPAD_CODEGEN_AUTO) {     \
      tds::OptimizationProblem<tds::DIFF_CPPAD_CODEGEN_AUTO,            \
                               NNBenchFunctor>::CostFunctor::Compile(); \
    }                                                                   \
    ProblemType problem;                                                \
    ProblemType::DoubleVector x(kParameterDim, 5.0);                    \
    problem.gradient(x);                                                \
    for (auto _ : state) {                                              \
      problem.gradient(x);                                              \
    }                                                                   \
  }                                                                     \
  BENCHMARK(BM_##diff_type##_NN_Grad);                                  \
  \
  static void BM_##diff_type##_LCPContactModel_Grad(benchmark::State& state) {       \
    using ProblemType =                                                 \
        tds::OptimizationProblem<tds::diff_type, LCPContactModelFunctor>;       \
    if constexpr (tds::diff_type == tds::DIFF_CPPAD_CODEGEN_AUTO) {     \
      tds::OptimizationProblem<tds::DIFF_CPPAD_CODEGEN_AUTO,            \
                               LCPContactModelFunctor>::CostFunctor::Compile(); \
    }                                                                   \
    ProblemType problem;                                                \
    ProblemType::DoubleVector x(kParameterDim, 5.0);                    \
    problem.gradient(x);                                                \
    for (auto _ : state) {                                              \
      problem.gradient(x);                                              \
    }                                                                   \
  }                                                                     \
  BENCHMARK(BM_##diff_type##_LCPContactModel_Grad);                                  \
                                                                        \
  static void BM_##diff_type##_Pendulum_Grad(benchmark::State& state) { \
    if constexpr (tds::diff_type == tds::DIFF_CPPAD_CODEGEN_AUTO) {     \
      std::cout << "Compiling...\n";                                    \
      tds::OptimizationProblem<tds::DIFF_CPPAD_CODEGEN_AUTO,            \
                               PendulumCost>::CostFunctor::Compile();   \
      std::cout << "Compiled.\n";                                       \
    }                                                                   \
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
  TEST(ADTest, NN_##diff_type##_VS_CERES) {                                  \
    std::default_random_engine e(1234);                                      \
    std::normal_distribution<> normal_dist(0, 1);                            \
    using DiffProblemType =                                                  \
        tds::OptimizationProblem<tds::diff_type, NNBenchFunctor>;            \
    using CeresProblemType =                                                 \
        tds::OptimizationProblem<tds::DIFF_CERES, NNBenchFunctor>;           \
    if constexpr (tds::diff_type == tds::DIFF_CPPAD_CODEGEN_AUTO) {          \
      DiffProblemType::CostFunctor::Compile();                               \
    }                                                                        \
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
  }                                                                          \
                                                                             \
  TEST(ADTest, Pendulum_##diff_type##_VS_CERES) {                            \
    std::default_random_engine e(1234);                                      \
    std::normal_distribution<> normal_dist(0, 1);                            \
    auto dproblem = create_problem<tds::diff_type>();                        \
    auto cproblem = create_problem<tds::DIFF_CERES>();                       \
    using DiffProblemType = decltype(dproblem);                              \
    if constexpr (tds::diff_type == tds::DIFF_CPPAD_CODEGEN_AUTO) {          \
      DiffProblemType::CostFunctor::Compile();                               \
    }                                                                        \
    using CeresProblemType = decltype(cproblem);                             \
    DiffProblemType::DoubleVector dx(DiffProblemType::kParameterDim);        \
    CeresProblemType::DoubleVector cx(CeresProblemType::kParameterDim);      \
    for (int trial = 0; trial < kTrials; ++trial) {                          \
      for (int i = 0; i < DiffProblemType::kParameterDim; ++i) {             \
        const double xi = normal_dist(e);                                    \
        dx[i] = xi;                                                          \
        cx[i] = xi;                                                          \
      }                                                                      \
      auto dgrad = dproblem.gradient(dx);                                    \
      auto cgrad = cproblem.gradient(cx);                                    \
      EXPECT_THAT(dgrad, ::testing::Pointwise(::testing::FloatEq(), cgrad)); \
    }                                                                        \
  }
