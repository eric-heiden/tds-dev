#include <gtest/gtest.h>

// clang-format off
// Differentiation must go first
#include "utils/differentiation.hpp"
#include "math/conditionals.hpp"
#include "world.hpp"
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
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

template <typename Algebra>
struct MatrixInverseFunctor {
  static const inline int kDim = 5;
  using Scalar = typename Algebra::Scalar;
  using MatrixX = typename Algebra::MatrixX;

  Scalar operator()(const std::vector<Scalar>& x) const {
    MatrixX mat(5, 5);

    for (int i = 0; i < 5; ++i) {
      for (int j = i; j < 5; ++j) {
        // just some "random-looking" numbers
        mat(i, j) = Algebra::from_double(0.05103 * ((i % 3 - 1 + j) % 7));
        mat(j, i) = mat(i, j);
      }
    }
    for (int i = 0; i < 5; ++i) {
      mat(i, i) = x[i];
    }
    MatrixX inv_mat(5, 5);
    bool pd = Algebra::symmetric_inverse(mat, inv_mat);
    assert(pd);
    Scalar sum = Algebra::zero();
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j) {
        sum += mat(i, j);
      }
    }
    return sum;
  }
};

// template <typename Algebra>
// struct SimpleContactFunctor {
//   static const inline int kDim = 3;
//   using Scalar = typename Algebra::Scalar;

//   Scalar operator()(const std::vector<Scalar>& x) const {
//     tds::World<Algebra> world;
//     world.create_multi_body
//   }
// };

TEST(CppAdCogeGen, ConditionalExpressions) {
  typedef tds::GradientFunctional<tds::DIFF_CPPAD_CODEGEN_AUTO,
                                  FunctorWithConditions>
      GradFun;
  GradFun::Compile();
  GradFun f;
  double v = f.value({1., 2., 3.});
  std::cout << "f([1,2,3]) = " << v << std::endl;
  EXPECT_NEAR(v, 2., 1e-9);
  const auto& grad = f.gradient({1., 2., 3.});
  std::cout << "d/dx f([1,2,3]) = [" << grad[0] << ", " << grad[1] << ", "
            << grad[2] << "]" << std::endl;
}

TEST(CppAdCogeGen, MatrixInverse) {
  typedef tds::GradientFunctional<tds::DIFF_CPPAD_CODEGEN_AUTO,
                                  MatrixInverseFunctor>
      CGFun;
  typedef tds::GradientFunctional<tds::DIFF_CERES, MatrixInverseFunctor>
      CeresFun;
  CGFun::Compile();
  CGFun f_cg;
  double v_cg = f_cg.value({1., 2., 3., 4., 5.});
  std::cout << "f_cg([1,2,3,4,5]) = " << v_cg << std::endl;
  CGFun f_ceres;
  double v_ceres = f_ceres.value({1., 2., 3., 4., 5.});
  std::cout << "f_ceres([1,2,3,4,5]) = " << v_ceres << std::endl;
  EXPECT_NEAR(v_cg, v_ceres, 1e-9);
  const auto& grad_cg = f_cg.gradient({1., 2., 3., 4., 5.});
  std::cout << "d/dx f_cg([1,2,3,4,5]) = [" << grad_cg[0] << ", " << grad_cg[1]
            << ", " << grad_cg[2] << ", " << grad_cg[3] << ", " << grad_cg[4]
            << "]" << std::endl;
  const auto& grad_ceres = f_ceres.gradient({1., 2., 3., 4., 5.});
  std::cout << "d/dx f_ceres([1,2,3,4,5]) = [" << grad_ceres[0] << ", "
            << grad_ceres[1] << ", " << grad_ceres[2] << ", " << grad_ceres[3]
            << ", " << grad_ceres[4] << "]" << std::endl;
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(grad_cg[i], grad_ceres[i], 1e-9);
  }
}