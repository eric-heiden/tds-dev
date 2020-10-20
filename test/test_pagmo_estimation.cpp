#include <gtest/gtest.h>

#include <thread>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>

#include "estimation_utils.hpp"
#include "utils/optim_gd.hpp"

constexpr double kEpsilon = 1e-5;

bool sequential_mode = true;
bool in_parallel() { return !sequential_mode; }
std::size_t thread_id() { return std::hash<std::thread::id>{}(std::this_thread::get_id()); }

TEST(PagmoEstimation, GradientDescent) {
  // CppAD::thread_alloc::parallel_setup(3, in_parallel, thread_id);
  // CppAD::thread_alloc::hold_memory(true);
  // CppAD::parallel_ad<double>();
  // sequential_mode = false;
  // auto problem = create_problem<tds::DIFF_CPPAD_AUTO>();
  auto problem =  create_problem<tds::DIFF_CPPAD_CODEGEN_AUTO>();
  problem.cost().Compile();
  pagmo::problem prob(problem);
  pagmo::algorithm algo{tds::optim_gd()};
  pagmo::archipelago archi{3u, algo, prob, 3u};
  archi.evolve(10);
  archi.wait_check();
  int i = 0;
  for (const auto& island : archi) {
    auto params = island.get_population().champion_x();
    double f = island.get_population().champion_f()[0];
    std::cout << "Best parameters from island " << (i++) << ": ";
    for (std::size_t j = 0; j < 2; ++j) {
      std::cout << params[j] << " ";
      // OptimLib does not achieve such high accuracy with the predefined
      // settings
      EXPECT_NEAR(params[j], true_link_lengths[j], 1e-2);
    }
    std::cout << "\terror: " << f << std::endl;
  }
}

TEST(PagmoEstimation, Sade) {
  pagmo::problem prob(create_problem<tds::DIFF_NUMERICAL>());
  pagmo::algorithm algo{pagmo::sade(100)};
  pagmo::archipelago archi{16u, algo, prob, 20u};
  archi.evolve(10);
  archi.wait_check();
  int i = 0;
  for (const auto& island : archi) {
    auto params = island.get_population().champion_x();
    double f = island.get_population().champion_f()[0];
    std::cout << "Best parameters from island " << (i++) << ": ";
    for (std::size_t j = 0; j < 2; ++j) {
      std::cout << params[j] << " ";
      EXPECT_NEAR(params[j], true_link_lengths[j], kEpsilon);
    }
    std::cout << "\terror: " << f << std::endl;
  }
}

TEST(PagmoEstimation, CeresNlopt) {
  pagmo::problem prob(create_problem<tds::DIFF_CERES>());
  pagmo::algorithm algo{pagmo::nlopt()};
  pagmo::archipelago archi{16u, algo, prob, 20u};
  archi.evolve(10);
  archi.wait_check();
  int i = 0;
  for (const auto& island : archi) {
    auto params = island.get_population().champion_x();
    double f = island.get_population().champion_f()[0];
    std::cout << "Best parameters from island " << (i++) << ": ";
    for (std::size_t j = 0; j < 2; ++j) {
      std::cout << params[j] << " ";
      EXPECT_NEAR(params[j], true_link_lengths[j], kEpsilon);
    }
    std::cout << "\terror: " << f << std::endl;
  }
}