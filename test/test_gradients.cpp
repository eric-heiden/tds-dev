#include <gtest/gtest.h>

#include "utils/differentiation.hpp"

const double kEpsilon = 1e-6;

template <typename Scalar>
struct l2 {
  Scalar operator()(const std::vector<Scalar>& vs) const {
    Scalar n(0.);
    for (const Scalar& v : vs) {
      n += v * v;
    }
    return n;
  }
};

TEST(Differentiation, FiniteDiffSimple) {
  std::vector<double> x{0.2, -0.5, 0.1, 123.45};
  std::vector<double> grad;
  tds::compute_gradient<tds::DIFF_NUMERICAL>(l2<double>(), x, grad);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(grad[i], 2. * x[i], kEpsilon);
  }
}

TEST(Differentiation, CeresSimple) {
  std::vector<double> x{0.2, -0.5, 0.1, 123.45};
  std::vector<double> grad;
  const int kDim = 4;
  tds::compute_gradient<tds::DIFF_CERES, kDim, l2>(x, grad);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(grad[i], 2. * x[i], kEpsilon);
  }
}

TEST(Differentiation, DualSimple) {
  std::vector<double> x{0.2, -0.5, 0.1, 123.45};
  std::vector<double> grad;
  tds::compute_gradient<tds::DIFF_DUAL>(l2<TinyDual<double>>(), x, grad);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(grad[i], 2. * x[i], kEpsilon);
  }
}
