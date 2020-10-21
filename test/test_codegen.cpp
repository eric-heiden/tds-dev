#include <gtest/gtest.h>

// clang-format off
// Differentiation must go first
#include "utils/differentiation.hpp"
#include "math/conditionals.hpp"
// clang-format on

template <typename Algebra>
struct FunctorWithConditions {
  static const inline int kDim = 3;
  using Scalar = typename Algebra::Scalar;

  Scalar operator()(const std::vector<Scalar>& x) const {
    Scalar y = Algebra::sin(x[0]);
    // return CppAD::CondExpGt(y, Algebra::zero(), Algebra::one(),
    //                         Algebra::zero());
    Scalar c1 =
        tds::where_gt(y, Algebra::zero(), Algebra::one(), Algebra::zero());
    Scalar c2 = tds::where_ge(x[1], Algebra::zero(), c1, Algebra::sqrt(c1));
    return c1 + c2;
  }
};

TEST(CppAdCogeGen, ConditionalExpressions) {
  typedef tds::GradientFunctional<tds::DIFF_CPPAD_CODEGEN_AUTO,
                                  FunctorWithConditions>
      GradFun;
  GradFun::Compile();
  GradFun f;
  double v = f.value({1., 2., 3.}) ;
  std::cout << "f([1,2,3]) = " << v << std::endl;
  EXPECT_NEAR(v, 2., 1e-9);
  const auto& grad = f.gradient({1., 2., 3.});
  std::cout << "d/dx f([1,2,3]) = [" << grad[0] << ", " << grad[1] << ", "
            << grad[2] << "]" << std::endl;
}