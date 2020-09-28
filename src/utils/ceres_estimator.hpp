/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <mutex>
#include <random>
#include <string>
#include <thread>

// clang-format off
#include "differentiation.hpp"
#include "math/eigen_algebra.hpp"
#include "parameter.hpp"

#include <ceres/ceres.h>
// clang-format on

// #define USE_MATPLOTLIB 1

#ifdef USE_MATPLOTLIB
#include "third_party/matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

namespace tds {
template <DiffMethod Method, template <typename> typename F>
class CeresEstimator : ceres::IterationCallback {
 public:
  static const int kParameterDim = F<EigenAlgebra>::kDim;
  static_assert(kParameterDim >= 1);

  std::array<EstimationParameter, kParameterDim> parameters;

  void set_params(const std::array<double, kParameterDim> &params) {
    for (int i = 0; i < kParameterDim; ++i) {
      parameters[i].value = params[i];
    }
  }

  /**
   * Whether to set parameter bounds (line search optimizers do not support
   * this).
   */
  bool set_bounds{true};

  ceres::Solver::Options options;

  CeresEstimator() {
    options.minimizer_progress_to_stdout = true;
    options.callbacks.push_back(this);
  }

 private:
  GradientFunctional<Method, F> cost_;
  struct CostFunctor : ceres::SizedCostFunction<1, kParameterDim> {
    CeresEstimator *parent{nullptr};
    GradientFunctional<Method, F> cost;

    CostFunctor(CeresEstimator *parent) : parent(parent) {}

    bool Evaluate(double const *const *parameters, double *residuals,
                  double **jacobians) const override {
      std::vector<double> x(*parameters, *parameters + kParameterDim);
      for (int i = 0; i < kParameterDim; ++i) {
        parent->current_param_[i] = x[i];
      }
      if (residuals != nullptr) {
        *residuals = cost.value(x);
      }
      if (jacobians != nullptr) {
        const std::vector<double> &gradient = cost.gradient(x);
        for (int i = 0; i < kParameterDim; ++i) {
          (*jacobians)[i] = gradient[i];
        }
      }
      return true;
    }
  };

 public:
  virtual ceres::Problem &setup(ceres::LossFunction *loss_function = nullptr) {
    if (vars_) {
      delete[] vars_;
    }
    if (cost_function_) {
      delete cost_function_;
    }
    vars_ = new double[kParameterDim];
    cost_function_ = new CostFunctor(this);
    problem_.AddResidualBlock(cost_function_, loss_function, vars_);

    for (int i = 0; i < kParameterDim; ++i) {
      vars_[i] = parameters[i].value;
    }
    if (set_bounds) {
      for (int i = 0; i < kParameterDim; ++i) {
        problem_.SetParameterLowerBound(vars_, i, parameters[i].minimum);
        problem_.SetParameterUpperBound(vars_, i, parameters[i].maximum);
      }
    }

    return problem_;
  }

  const GradientFunctional<Method, F> &cost() const { return cost_; }

  void gradient_descent(double learning_rate, int iterations) {
    double cost;
    param_evolution_.clear();
    std::vector<double> x(vars_, vars_ + kParameterDim);
    for (int i = 0; i < iterations; ++i) {
      double cost = cost_.value(x);
      const auto &gradient = cost_.gradient(x);
      printf("Gradient descent step %i - cost: %.6f\n", i, cost);
      for (int j = 0; j < kParameterDim; ++j) {
        current_param_[j] = x[j];
        x[j] -= learning_rate * gradient[j];
      }
      param_evolution_.push_back(current_param_);
    }
    for (int i = 0; i < kParameterDim; ++i) {
      parameters[i].value = x[i];
    }
  }

  ceres::Solver::Summary solve() {
    ceres::Solver::Summary summary;
    param_evolution_.clear();
    best_cost_ = std::numeric_limits<double>::max();
    for (int i = 0; i < kParameterDim; ++i) {
      vars_[i] = parameters[i].value;
      best_params_[i] = parameters[i].value;
    }
    ceres::Solve(options, &problem_, &summary);
    if (summary.final_cost > best_cost_) {
      printf(
          "Ceres returned a parameter vector with a final cost of %.8f whereas "
          "during the optimization a parameter vector with a lower cost of "
          "%.8f was found. Returning the best parameter vector.\n",
          summary.final_cost, best_cost_);
      for (int i = 0; i < kParameterDim; ++i) {
        parameters[i].value = best_params_[i];
      }
    } else {
      for (int i = 0; i < kParameterDim; ++i) {
        parameters[i].value = vars_[i];
      }
    }
    return summary;
  }

  const double *vars() const { return vars_; }

  virtual ~CeresEstimator() {
    if (vars_) {
      delete[] vars_;
      vars_ = nullptr;
    }
  }

  const std::vector<std::array<double, kParameterDim>> &parameter_evolution()
      const {
    return param_evolution_;
  }

  double best_cost() const { return best_cost_; }

  const std::array<double, kParameterDim> &best_parameters() const {
    return best_params_;
  }

 private:
  ceres::Problem problem_;
  double *vars_{nullptr};
  CostFunctor *cost_function_{nullptr};
  std::vector<std::array<double, kParameterDim>> param_evolution_;
  mutable std::array<double, kParameterDim> current_param_;

  mutable double best_cost_{std::numeric_limits<double>::max()};
  mutable std::array<double, kParameterDim> best_params_;

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary &summary) override {
    param_evolution_.push_back(current_param_);
    return ceres::SOLVER_CONTINUE;
  }
};  // namespace tds

/**
 * Implements Parallel Basin Hopping that combines local gradient-based
 * optimization using Ceres with random guessing to overcome poor local
 * optima.
 *
 * McCarty & McGuire "Parallel Monotonic Basin Hopping for Low Thrust
 * Trajectory Optimization"
 */
template <int ParameterDim, typename Estimator>
class BasinHoppingEstimator {
  static const int kParameterDim = ParameterDim;
  typedef std::function<std::unique_ptr<Estimator>()> EstimatorConstructor;

 public:
  EstimatorConstructor estimator_constructor;
  std::array<double, kParameterDim> params;
  std::size_t num_workers;

  /**
   * Time limit in seconds.
   */
  double time_limit{1.0};

  /**
   * Terminate if estimation cost drops below this value.
   */
  double cost_limit{1e-3};

  /**
   * Initial standard deviation used for Gaussian noise applied to the
   * parameters, normalized by the bounds of the parameter.
   */
  double initial_std{1.};

  /**
   * Whether to reduce the standard deviation of the random guess in the
   * parameter as the iteration count increases.
   */
  bool fade_std{false};

  BasinHoppingEstimator(
      const EstimatorConstructor &estimator_constructor,
      const std::array<double, kParameterDim> &initial_guess,
      std::size_t num_workers = std::thread::hardware_concurrency())
      : estimator_constructor(estimator_constructor),
        params(initial_guess),
        num_workers(num_workers) {
    workers_.reserve(num_workers);
  }

  // Run global optimizer.
  void run() {
    using namespace std::chrono;
    best_cost_ = std::numeric_limits<double>::max();
    std::cout << "Starting " << num_workers << " worker(s).\n";
    auto start_time = high_resolution_clock::now();
    for (std::size_t k = 0; k < num_workers; ++k) {
      workers_.emplace_back([this, k, &start_time]() {
        std::seed_seq seed{
            // Time
            static_cast<std::size_t>(std::chrono::high_resolution_clock::now()
                                         .time_since_epoch()
                                         .count()),
            // counter
            k};

        std::mt19937 eng(seed);

        auto estimator = this->estimator_constructor();
        estimator->setup(new ceres::TrivialLoss);  // new ceres::HuberLoss(1.));
                                                   // // TODO expose this
        if (k == 0) {
          // set initial guess
          estimator->set_params(this->params);
        } else {
          for (auto &p : estimator->parameters) {
            p = p.random_value();
          }
        }
        for (int iter = 0;; ++iter) {
          {
            std::unique_lock<std::mutex> lock(this->mutex_);
            // check stopping criteria
            auto stop_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop_time - start_time);
            bool time_up = static_cast<long>(duration.count()) >=
                           static_cast<long>(time_limit * 1e3);
#ifdef DEBUG
            if (time_up) {
              std::cout << "time up\n";
            }
            if (this->best_cost_ < this->cost_limit) {
              std::cout << "this->best_cost_ < this->cost_limit\n";
            }
            if (this->stop_) {
              std::cout << "this->stop_\n";
            }
#endif
            if (time_up || this->stop_ || this->best_cost_ < this->cost_limit) {
              std::cout << "Thread " << k << " has terminated after " << iter
                        << " iterations. ";
              printf("time up? %d  stop? %d  best cost? %d\n", time_up,
                     this->stop_, this->best_cost_ < this->cost_limit);
              return;
            }
          }
          double &solver_time_limit =
              estimator->options.max_solver_time_in_seconds;
          if (solver_time_limit > time_limit) {
            solver_time_limit = time_limit;
          }
          auto summary = estimator->solve();
          std::cout << summary.FullReport() << std::endl;
          {
            std::unique_lock<std::mutex> lock(this->mutex_);
            if (estimator->best_cost() < this->best_cost_) {
              this->best_cost_ = estimator->best_cost();
              printf("FOUND NEW BEST COST: %.6f\n", estimator->best_cost());
              for (int i = 0; i < kParameterDim; ++i) {
                this->params[i] = estimator->parameters[i].value;
              }
            }
          }
          // apply random change to the parameters
          for (int i = 0; i < kParameterDim; ++i) {
            auto &param = estimator->parameters[i];
            if (fade_std) {
              std::normal_distribution<double> d{
                  this->params[i],
                  initial_std / (iter + 1.) * (param.maximum - param.minimum)};
              param.value = d(eng);
            } else {
              std::normal_distribution<double> d{
                  this->params[i],
                  initial_std * (param.maximum - param.minimum)};
              param.value = d(eng);
            }
            param.value = std::max(param.minimum, param.value);
            param.value = std::min(param.maximum, param.value);
          }
        }
      });
    }
    for (auto &worker : workers_) {
      worker.join();
    }
    workers_.clear();
  }

  void stop() {
    std::lock_guard<std::mutex> lock{mutex_};
    stop_ = true;
  }

  virtual ~BasinHoppingEstimator() {
    stop();
    for (auto &worker : workers_) {
      worker.join();
    }
  }

  double best_cost() const { return best_cost_; }

 private:
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  bool stop_{false};

  double best_cost_;
};
}  // namespace tds
