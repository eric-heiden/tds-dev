// clang-format off
#include "utils/differentiation.hpp"
#include "utils/cuda_codegen.hpp"
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "math/tiny/tiny_algebra.hpp"
#include "urdf/urdf_cache.hpp"
#include "utils/stopwatch.hpp"
#include "visualizer/opengl/visualizer.h"
<<<<<<< HEAD
// clang-format on
=======
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>


// clang-format on
using namespace TINY;
>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83

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

  int input_dim() const { return system->dof() + system->dof_qd(); }
  int state_dim() const {
    return system->dof() + system->dof_qd() + system->size() * 3;
  }
  int output_dim() const { return num_timesteps * state_dim(); }

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
    for (int i = 0; i < system->dof_qd(); ++i) {
      system->qd(i) = v[i + system->dof()];
    }
    std::vector<Scalar> result(output_dim());
    for (int t = 0; t < num_timesteps; ++t) {
      tds::forward_dynamics(*system, world.get_gravity());
      system->clear_forces();
      world.step(dt);
      tds::integrate_euler(*system, dt);
      int j = 0;
      for (int i = 0; i < system->dof(); ++i, ++j) {
        result[j] = system->q(i);
      }
      for (int i = 0; i < system->dof_qd(); ++i, ++j) {
        result[j] = system->qd(i);
      }
      for (const auto link : *system) {
        result[j++] = link.X_world.translation[0];
        result[j++] = link.X_world.translation[1];
        result[j++] = link.X_world.translation[2];
      }
    }
    return result;
  }
};

<<<<<<< HEAD
=======
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());

    return result;
}



>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83
int main(int argc, char* argv[]) {
  using Scalar = double;
  using CGScalar = typename CppAD::cg::CG<Scalar>;
  using Dual = typename CppAD::AD<CGScalar>;

  using DiffAlgebra =
      tds::default_diff_algebra<tds::DIFF_CPPAD_CODEGEN_AUTO, 0, Scalar>::type;

  ContactSimulation<DiffAlgebra> simulation;

  // trace function with all zeros as input
  std::vector<Dual> ax(simulation.input_dim(), Dual(0));
  CppAD::Independent(ax);
  std::vector<Dual> ay(simulation.output_dim());
  std::cout << "Tracing function for code generation...\n";
  ay = simulation(ax);
  CppAD::ADFun<CGScalar> tape;
  tape.Dependent(ax, ay);
  tape.optimize();

  tds::Stopwatch timer;
  timer.start();
  std::string model_name = "cuda_model";
  tds::CudaSourceGen<Scalar> cgen(tape, model_name);
  cgen.setCreateForwardZero(true);

  tds::CudaLibraryProcessor p(&cgen);
<<<<<<< HEAD
=======
  std::string nvcc_path = exec("which nvcc");
  std::cout << "Using ["<< nvcc_path << "]" << std::endl;
  p.nvcc_path()=nvcc_path;
>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83
  p.generate_code();
  p.create_library();

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
<<<<<<< HEAD
  int num_total_threads = 400;
=======
  int num_total_threads = 2048;
>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83

  std::vector<std::vector<Scalar>> outputs(
      num_total_threads, std::vector<Scalar>(simulation.output_dim()));

  std::vector<std::vector<Scalar>> inputs(num_total_threads);

  TinyOpenGL3App app("CUDA Pendulum", 1024, 768);
  app.m_renderer->init();
<<<<<<< HEAD
  app.set_up_axis(2);
=======
  int upAxis = 2;
  app.set_up_axis(upAxis);
>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83
  app.m_renderer->get_active_camera()->set_camera_distance(4);
  app.m_renderer->get_active_camera()->set_camera_pitch(-30);
  app.m_renderer->get_active_camera()->set_camera_target_position(0, 0, 0);
  // install ffmpeg in path and uncomment, to enable video recording
  // app.dump_frames_to_video("test.mp4");

  int sphere_shape = app.register_graphics_unit_sphere_shape(SPHERE_LOD_LOW);
<<<<<<< HEAD
  typedef tds::Conversion<DiffAlgebra, tds::TinyAlgebraf> Conversion;
=======
  //typedef tds::Conversion<DiffAlgebra, tds::TinyAlgebraf> Conversion;
>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83
  std::vector<int> visuals;
  for (int i = 0; i < num_total_threads * simulation.system->size(); ++i) {
    TinyVector3f pos(0, 0, 0);
    TinyQuaternionf orn(0, 0, 0, 1);
    TinyVector3f color(0.5, 0.6, 1);
    TinyVector3f scaling(0.1, 0.1, 0.1);
    int instance = app.m_renderer->register_graphics_instance(
        sphere_shape, pos, orn, color, scaling);
    visuals.push_back(instance);
  }

  model.forward_zero.allocate(num_total_threads);

  TinyVector3f line_color(0.3, 0.3, 0.3);
  float line_width = 1;
  const int link_pos_id_offset =
      simulation.system->dof() + simulation.system->dof_qd();
  const int square_id = (int)std::sqrt((double)num_total_threads);
  const float radius = 1.f;
  for (int run = 0; run < 40; ++run) {
    for (int i = 0; i < num_total_threads; ++i) {
      inputs[i] = std::vector<Scalar>(simulation.input_dim(), Scalar(0));
      inputs[i][0] = std::rand() * 1. / RAND_MAX;
    }
    for (int t = 0; t < 1000; ++t) {
      timer.start();
      // call GPU kernel
      model.forward_zero(&outputs, inputs, 64);
      timer.stop();
      std::cout << "Kernel execution took " << timer.elapsed() << " seconds.\n";

      for (int i = 0; i < num_total_threads; ++i) {
        for (int j = 0; j < simulation.input_dim(); ++j) {
          inputs[i][j] = outputs[i][j];
        }
      }

      int visual_id = 0;
      for (int i = 0; i < num_total_threads; ++i) {
        TinyVector3f prev_pos(0, 0, 0);
        for (std::size_t l = 0; l < simulation.system->size(); ++l) {
          int sphereId = visuals[visual_id++];
          TinyVector3f pos;
          pos[0] = outputs[i][link_pos_id_offset + l * 3 + 0];
          pos[1] = outputs[i][link_pos_id_offset + l * 3 + 1];
          pos[2] = outputs[i][link_pos_id_offset + l * 3 + 2];
          pos[0] += radius * (i % square_id) - square_id * radius / 2;
          pos[1] += radius * (i / square_id) - square_id * radius / 2;
          TinyQuaternionf orn(0, 0, 0, 1);
          if (l > 0) {
<<<<<<< HEAD
            app.m_renderer->draw_line(prev_pos, pos, line_color, line_width);
=======
            //app.m_renderer->draw_line(prev_pos, pos, line_color, line_width);
>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83
          }
          prev_pos = pos;
          app.m_renderer->write_single_instance_transform_to_cpu(pos, orn,
                                                                 sphereId);
        }
      }
<<<<<<< HEAD
      DrawGridData data;
      data.upAxis = 2;
=======
      app.m_renderer->update_camera(upAxis);
    
      DrawGridData data;
      data.drawAxis = true;
      data.upAxis = upAxis;
>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83
      app.draw_grid(data);
      app.m_renderer->render_scene();
      app.m_renderer->write_transforms();
      app.swap_buffer();
    }
  }

  model.forward_zero.deallocate();

  for (const auto& thread : outputs) {
    for (const Scalar& t : thread) {
      std::cout << t << "  ";
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
<<<<<<< HEAD
}
=======
}
>>>>>>> 776e2b158d51d830026cc391f0f4ff2370817f83
