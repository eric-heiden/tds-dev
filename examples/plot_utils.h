#ifndef PLOT_UTILS_H
#define PLOT_UTILS_H

#include <string>
#include <vector>

#include "matplotlib-cpp.h"
#include "tiny_double_utils.h"

namespace plt = matplotlibcpp;

template <typename T, typename TUtils>
static void plot_trajectory(const std::vector<std::vector<T>> &states,
                            const std::string &title = "Figure") {
  typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils,
                             CeresUtils<kParameterDim>>
      Utils;
  for (int i = 0; i < static_cast<int>(states[0].size()); ++i) {
    std::vector<double> traj(states.size());
    for (int t = 0; t < static_cast<int>(states.size()); ++t) {
      traj[t] = Utils::getDouble(states[t][i]);
    }
    plt::named_plot("state[" + std::to_string(i) + "]", traj);
  }
  plt::legend();
  plt::title(title);
  plt::show();
}

#endif
