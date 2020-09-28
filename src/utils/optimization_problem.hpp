#pragma once

#if USE_PAGMO
#include <pagmo/types.hpp>
#endif

#include "differentiation.hpp"
#include "parameter.hpp"

namespace tds {

template <DiffMethod Method, template <typename> typename F>
class OptimizationProblem {
 public:
  static const int kParameterDim = F<EigenAlgebra>::kDim;
  static_assert(kParameterDim >= 1);
  typedef std::array<EstimationParameter, kParameterDim> ParameterVector;
  typedef GradientFunctional<Method, F> CostFunctor;

#if USE_PAGMO
  typedef pagmo::vector_double DoubleVector;
#else
  typedef std::vector<double> DoubleVector;
#endif

  const unsigned int m_dim = kParameterDim;

 protected:
  CostFunctor cost_;
  ParameterVector parameters_;

 public:
  ParameterVector& parameters() { return parameters_; }
  const ParameterVector& parameters() const { return parameters_; }

  EstimationParameter& operator[](int i) { return parameters_[i]; }
  const EstimationParameter& operator[](int i) const { return parameters_[i]; }

  CostFunctor& cost() { return cost_; }
  const CostFunctor& cost() const { return cost_; }

  void set_params(const std::array<double, kParameterDim>& params) {
    for (int i = 0; i < kParameterDim; ++i) {
      parameters[i].value = params[i];
    }
  }

  OptimizationProblem() = default;
  OptimizationProblem(OptimizationProblem&) = default;
  OptimizationProblem(const OptimizationProblem&) = default;
  OptimizationProblem(const ParameterVector& parameters)
      : parameters_(parameters) {}
  OptimizationProblem& operator=(const OptimizationProblem&) = default;

  virtual ~OptimizationProblem() = default;

  std::pair<DoubleVector, DoubleVector> get_bounds() const {
    DoubleVector low(kParameterDim), high(kParameterDim);
    for (int i = 0; i < kParameterDim; ++i) {
      low[i] = parameters_[i].minimum;
      high[i] = parameters_[i].maximum;
    }
    return {low, high};
  }

  DoubleVector fitness(const DoubleVector& x) const { return {cost_.value(x)}; }

  DoubleVector gradient(const DoubleVector& x) const {
    return cost_.gradient(x);
  }

  bool has_gradient() const { return true; }
};
}  // namespace tds