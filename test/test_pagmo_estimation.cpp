#include <gtest/gtest.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>

#include "estimation_utils.hpp"
#include "utils/optim_gd.hpp"

constexpr double kEpsilon = 1e-5;

TEST(PagmoEstimation, GradientDescent) {
  pagmo::problem prob(create_problem<tds::DIFF_CERES>());
  pagmo::algorithm algo{tds::optim_gd()};
  pagmo::archipelago archi{1u, algo, prob, 1u}; // Proof of concept
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