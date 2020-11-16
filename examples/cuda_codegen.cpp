// clang-format off
#include "utils/differentiation.hpp"
#include "utils/cuda_codegen.hpp"
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "math/tiny/tiny_algebra.hpp"
#include "urdf/urdf_cache.hpp"
#include "utils/stopwatch.hpp"
// clang-format on

// function to be converted to CUDA code
template <typename Algebra>
struct MyFunction {
  using Scalar = typename Algebra::Scalar;

  int input_dim() const { return 2; }
  int output_dim() const { return 2; }

  std::vector<Scalar> operator()(const std::vector<Scalar>& v) {
    std::vector<Scalar> result(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      const auto& e = v[i];
      result[i] = e * e;
      result[i] += tds::where_ge(e, Scalar(0), e * e, e * e * e);
    }
    return result;
  }
};

template <typename Algebra>
struct ContactSimulation {
  using Scalar = typename Algebra::Scalar;
  tds::UrdfCache<Algebra> cache;

  tds::World<Algebra> world;
  tds::MultiBody<Algebra>* system = nullptr;

  int num_timesteps{1};
  Scalar dt{Algebra::from_double(3e-3)};

  int input_dim() const { return system->dof(); }
  int output_dim() const { return num_timesteps * system->dof(); }

  ContactSimulation() {
    std::string plane_filename, urdf_filename;
    tds::FileUtils::find_file("plane_implicit.urdf", plane_filename);
    tds::FileUtils::find_file("pendulum5.urdf", urdf_filename);
    cache.construct(plane_filename, world, false, false);
    system = cache.construct(urdf_filename, world, false, false);
    system->base_X_world().translation = Algebra::unit3_z();
  }

  std::vector<Scalar> operator()(const std::vector<Scalar>& v) {
    assert(static_cast<int>(v.size()) == input_dim());
    system->initialize();
    for (int i = 0; i < system->dof(); ++i) {
      system->q(i) = v[i];
    }
    std::vector<Scalar> result(output_dim());
    for (int t = 0; t < num_timesteps; ++t) {
      tds::forward_dynamics(*system, world.get_gravity());
      system->clear_forces();
      world.step(dt);
      tds::integrate_euler(*system, dt);
      for (int i = 0; i < system->dof(); ++i) {
        result[t * system->dof() + i] = system->q(i);
      }
    }
    return result;
  }
};

int main(int argc, char* argv[]) {
  using Scalar = double;
  using CGScalar = typename CppAD::cg::CG<Scalar>;
  using Dual = typename CppAD::AD<CGScalar>;

  using DiffAlgebra =
      tds::default_diff_algebra<tds::DIFF_CPPAD_CODEGEN_AUTO, 0, Scalar>::type;

  ContactSimulation<DiffAlgebra> fun;

  // trace function with all zeros as input
  std::vector<Dual> ax(fun.input_dim(), Dual(0));
  CppAD::Independent(ax);
  std::vector<Dual> ay(fun.output_dim());
  std::cout << "Tracing function for code generation...\n";
  ay = fun(ax);
  CppAD::ADFun<CGScalar> tape;
  tape.Dependent(ax, ay);
  tape.optimize();

  tds::Stopwatch timer;
  timer.start();
  std::string model_name = "cuda_model";
  tds::CudaSourceGen<Scalar> cgen(tape, model_name);
  cgen.setCreateForwardZero(true);

  tds::CudaLibraryProcessor p(&cgen);
  p.generate_code();
  // p.create_library();

  // CppAD::cg::ModelLibraryCSourceGen<Scalar> libcgen(cgen);
  // libcgen.setVerbose(true);
  // CppAD::cg::DynamicModelLibraryProcessor<Scalar> p(libcgen);
  // auto compiler = std::make_unique<CppAD::cg::ClangCompiler<Scalar>>();
  // compiler->setSourcesFolder("cgen_srcs");
  // compiler->setSaveToDiskFirst(true);
  // compiler->addCompileFlag("-O" + std::to_string(1));
  // p.setLibraryName(model_name);
  // p.createDynamicLibrary(*compiler, false);

  // create model to load shared library
  tds::CudaModel<Scalar> model(model_name);

  // how many threads to run on the GPU
  int num_total_threads = 5000;

  std::vector<std::vector<Scalar>> outputs(
      num_total_threads, std::vector<Scalar>(fun.output_dim()));
  // generate some fake inputs
  std::vector<std::vector<Scalar>> inputs(num_total_threads);
  for (int i = 0; i < num_total_threads; ++i) {
    inputs[i] = std::vector<Scalar>(fun.input_dim());
    for (int j = 0; j < fun.input_dim(); ++j) {
      inputs[i][j] = std::rand() * 1. / RAND_MAX;
    }
  }

  model.forward_zero.allocate(num_total_threads);

  // call GPU kernel
  for (int i = 0; i < 100; ++i) {
    timer.start();
    model.forward_zero(&outputs, inputs, 64);
    timer.stop();
    std::cout << "Kernel execution took " << timer.elapsed() << " seconds.\n";
  }

  model.forward_zero.deallocate();

  for (const auto& thread : outputs) {
    for (const Scalar& t : thread) {
      std::cout << t << "  ";
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}