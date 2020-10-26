#include <gtest/gtest.h>

// clang-format off
// Differentiation must go first
#include "utils/differentiation.hpp"
#include "math/conditionals.hpp"
#include "world.hpp"
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "urdf/system_constructor.hpp"
#include "urdf/urdf_cache.hpp"
#include "utils/file_utils.hpp"
#include "mb_constraint_solver_spring.hpp"
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

template <typename Algebra>
struct NaNFunctor {
  static const inline int kDim = 1;
  using Scalar = typename Algebra::Scalar;

  Scalar operator()(const std::vector<Scalar>& x) const {
    Scalar is_nan = tds::where_eq(x[0], x[0], Algebra::zero(), Algebra::one());
    return is_nan;
  }
};

TEST(CppAdCogeGen, NaNCheck) {
  typedef tds::GradientFunctional<tds::DIFF_CPPAD_CODEGEN_AUTO, NaNFunctor>
      GradFun;
  GradFun::Compile();
  GradFun f;
  double v = f.value({std::nan("!")});
  std::cout << "isnan? = " << v << std::endl;
  EXPECT_NEAR(v, 1., 1e-9);
  v = f.value({123.});
  std::cout << "isnan? = " << v << std::endl;
  EXPECT_NEAR(v, 0., 1e-9);
}

TEST(CppAdCogeGen, ContactModel) {
  typedef tds::GradientFunctional<tds::DIFF_CPPAD_CODEGEN_AUTO,
                                  SpringContactModelFunctor>
      CGFun;
  typedef tds::GradientFunctional<tds::DIFF_CERES, SpringContactModelFunctor>
      CeresFun;
  std::srand(123);
  std::vector<double> x;
  std::cout << "x = [";
  for (int i = 0; i < CGFun::kDim; ++i) {
    x.push_back(std::rand() / RAND_MAX * 2.);
    std::cout << " " << x.back();
  }
  std::cout << " ]\n";
  CGFun::Compile();
  CGFun f_cg;
  double v_cg = f_cg.value(x);
  std::cout << "f_cg(x) = " << v_cg << std::endl;
  CGFun f_ceres;
  double v_ceres = f_ceres.value(x);
  std::cout << "f_ceres(x) = " << v_ceres << std::endl;
  EXPECT_NEAR(v_cg, v_ceres, 1e-9);
  const auto& grad_cg = f_cg.gradient(x);
  std::cout << "d/dx f_cg(x) = [";
  for (int i = 0; i < CGFun::kDim; ++i) {
    std::cout << " " << grad_cg[i];
  }
  std::cout << "]" << std::endl;
  const auto& grad_ceres = f_ceres.gradient(x);
  std::cout << "d/dx f_ceres(x) = [";
  for (int i = 0; i < CGFun::kDim; ++i) {
    std::cout << " " << grad_ceres[i];
  }
  std::cout << "]" << std::endl;
  for (int i = 0; i < CGFun::kDim; ++i) {
    EXPECT_NEAR(grad_cg[i], grad_ceres[i], 1e-9);
  }
}

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