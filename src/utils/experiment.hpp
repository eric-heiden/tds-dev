#pragma once

#include <ctime>
#include <cxxopts.hpp>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

#if USE_PAGMO
#include <pagmo/algorithm.hpp>
// #include <pagmo/algorithms/ipopt.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>
#endif

#include "ceres_estimator.hpp"
#include "neural_augmentation.hpp"
#include "optimization_problem.hpp"

namespace tds {
class Experiment {
 protected:
  const std::string name;
  nlohmann::json log;

  NeuralAugmentation augmentation;

 public:
  Experiment(const std::string& name) : name(name) {
    log["name"] = name;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    log["created"] = oss.str();
    log["settings"]["optimizer"] = "pagmo";
    log["settings"]["pagmo"]["solver"] = "nlopt";
    log["settings"]["pagmo"]["nlopt"]["solver"] = "lbfgs";
    log["settings"]["pagmo"]["nlopt"]["xtol_rel"] = 1e-2;
    log["settings"]["pagmo"]["nlopt"]["xtol_abs"] = 1e-10;
    log["settings"]["pagmo"]["nlopt"]["verbosity"] = 1;
    log["settings"]["pagmo"]["nlopt"]["max_time"] = 10.;
    log["settings"]["pagmo"]["num_islands"] = 5;
    log["settings"]["pagmo"]["num_individuals"] = 7;
    log["settings"]["pagmo"]["num_evolutions"] = 20;
    log["parameter_evolution"] = {};
    log["settings"]["augmentation"]["input_lasso_regularization"] = 0.;
    log["settings"]["augmentation"]["upper_l2_regularization"] = 0.;
    log["settings"]["augmentation"]["weight_limit"] = 0.5;
    log["settings"]["augmentation"]["bias_limit"] = 0.5;
    log["settings"]["augmentation"]["augmentation_is_residual"] = true;
  }

  bool save_log(const std::string& filename) const {
    namespace fs = std::filesystem;
    std::ofstream file(filename);
    if (!file.good()) {
      std::cerr << "Failed to save experiment log at " << filename << std::endl;
      return false;
    }
    file << log.dump(2);
    file.close();
    std::string abs_filename = fs::canonical(fs::path(filename)).u8string();
    std::cout << "Saved experiment log at " << abs_filename << std::endl;
    return true;
  }

  void parse_settings(int argc, char* argv[]) {
    cxxopts::Options options(name);
    std::map<std::string, nlohmann::json*> index;
    get_options(log["settings"], &options, &index);
    try {
      auto result = options.parse(argc, argv);
      if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
      }
      for (auto& [key, value] : index) {
        // clang-format off
        switch (value->type()) {
          case nlohmann::detail::value_t::boolean:
            *value = result[key].as<bool>();
            break;
          case nlohmann::detail::value_t::number_float:
            *value = result[key].as<double>();
            break;
          case nlohmann::detail::value_t::number_integer:
            *value = result[key].as<int>();
            break;
          case nlohmann::detail::value_t::number_unsigned:
            *value = result[key].as<unsigned int>();
            break;
          case nlohmann::detail::value_t::string: default: 
            *value = result[key].as<std::string>();
            break;
        }
        // clang-format on
      }
    } catch (const cxxopts::OptionException& e) {
      std::cout << "error parsing options: " << e.what() << std::endl;
      std::exit(1);
    }
  }

  nlohmann::json& operator[](const std::string& key) { return log[key]; }
  const nlohmann::json& operator[](const std::string& key) const {
    return log[key];
  }

  virtual void after_iteration() {}

  template <typename OptimizationProblem>
  void run(const OptimizationProblem& problem) {
    if (log["settings"]["optimizer"] == "pagmo") {
      run_pagmo(problem);
      return;
    }
    if (log["settings"]["optimizer"] == "ceres") {
      run_ceres(problem);
      return;
    }
    std::cerr << "Invalid optimizer setting \"" << log["settings"]["optimizer"]
              << "\".\n";
    std::exit(1);
  }

 protected:
  template <typename OptimizationProblemT>
  void update_log(const OptimizationProblemT& problem) {
    log["param_dim"] = problem.kParameterDim;
    log["diff_method"] = tds::diff_method_name(problem.kDiffMethod);
    for (const auto& param : problem.parameters()) {
      log["params"][param.name] = param.value;
    }
  }

  template <typename OptimizationProblemT>
  void run_pagmo(const OptimizationProblemT& problem) {
    update_log(problem);
#if USE_PAGMO
    pagmo::problem prob(problem);
    pagmo::nlopt solver(log["settings"]["pagmo"]["nlopt"]["solver"]);
    solver.set_maxtime(
        log["settings"]["pagmo"]["nlopt"]["max_time"]);  // in seconds
    solver.set_verbosity(log["settings"]["pagmo"]["nlopt"]
                            ["verbosity"]);  // print every n function evals
    solver.set_xtol_abs(log["settings"]["pagmo"]["nlopt"]["xtol_abs"]);
    solver.set_xtol_rel(log["settings"]["pagmo"]["nlopt"]["xtol_rel"]);
#else
    throw std::runtime_error(
        "CMake option 'USE_PAGMO' needs to be active to use Pagmo.");
#endif
  }

  template <typename OptimizationProblemT>
  void run_ceres(const OptimizationProblemT& problem) {
    update_log(problem);
    tds::CeresEstimator estimator(&problem);
    estimator.setup();
    auto summary = estimator.solve();
    std::cout << summary.FullReport() << std::endl;
    // parameters = problem.parameters();
    log["settings"]["solver"]["name"] = "ceres-LM";
  }

  virtual void save_settings() {}

  // get JSON leaves and populate options with settings
  static void get_options(nlohmann::json& data, cxxopts::Options* options,
                          std::map<std::string, nlohmann::json*>* index,
                          const std::string& prefix = "") {
    if (!data.is_object()) return;

    std::string pref = prefix.empty() ? "" : prefix + "_";
    for (auto& [key, value] : data.items()) {
      if (value.is_primitive()) {
        switch (value.type()) {
          case nlohmann::detail::value_t::boolean:
            options->add_options()(prefix + key, "",
                                   cxxopts::value<bool>()->default_value(
                                       std::to_string((bool)value)));
            break;
          case nlohmann::detail::value_t::number_float:
            options->add_options()(prefix + key, "",
                                   cxxopts::value<double>()->default_value(
                                       std::to_string((double)value)));
            break;
          case nlohmann::detail::value_t::number_integer:
            options->add_options()(prefix + key, "",
                                   cxxopts::value<int>()->default_value(
                                       std::to_string((int)value)));
            break;
          case nlohmann::detail::value_t::number_unsigned:
            options->add_options()(
                prefix + key, "",
                cxxopts::value<unsigned int>()->default_value(
                    std::to_string((unsigned int)value)));
            break;
          case nlohmann::detail::value_t::string:
          default:
            options->add_options()(
                prefix + key, "",
                cxxopts::value<std::string>()->default_value(value));
            break;
        }
        (*index)[prefix + key] = &value;
      } else if (value.is_object()) {
        get_options(value, options, index, prefix + key);
      }
    }
  }
};
}  // namespace tds