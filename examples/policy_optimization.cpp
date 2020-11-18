// clang-format off
#include "utils/differentiation.hpp"  // differentiation.hpp has to go first
#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/kinematics.hpp"
#include "math/tiny/tiny_algebra.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "multi_body.hpp"
#include "utils/dataset.hpp"
#include "utils/experiment.hpp"
#include "utils/file_utils.hpp"
#include "visualizer/opengl/tiny_opengl3_app.h"
#include "utils/pendulum.hpp"
// clang-format on

using namespace TINY;
using namespace tds;

const int dof = 3;
const std::size_t num_layers = 2;  // TOTAL number of layers
const std::size_t num_hidden_units = 4;

constexpr StaticNeuralNetworkSpecification<num_layers> policy_nn_spec() {
  StaticNeuralNetworkSpecification<num_layers> augmentation;
  for (std::size_t i = 0; i < num_layers; ++i) {
    augmentation.layers[i] = num_hidden_units;
    augmentation.use_bias[i] = true;
    augmentation.activations[i] = tds::NN_ACT_ELU;
  }
  augmentation.layers.front() = dof;
  augmentation.layers.back() = dof;

  augmentation.use_bias.front() = true;
  augmentation.activations.back() = tds::NN_ACT_IDENTITY;
  return augmentation;
}

constexpr int param_dim = policy_nn_spec().num_parameters();

/**
 * Gym environment that computes the cost (i.e. opposite of reward) to be
 * minimized by the policy optimizer.
 */
template <typename Algebra>
struct CostFunctor {
  using Scalar = typename Algebra::Scalar;
  // XXX this is important for the AD libraries
  static const int kDim = param_dim;
  int timesteps{200};
  Scalar dt{Algebra::from_double(1e-2)};

  tds::World<Algebra> world;
  tds::MultiBody<Algebra> *system;

  mutable tds::NeuralNetwork<Algebra> policy;

  CostFunctor() : policy(policy_nn_spec()) {
    system = world.create_multi_body();
    init_compound_pendulum<Algebra>(*system, world, dof);
  }

  /**
   * Rollout function that, given the policy parameters x, computes the cost.
   */
  Scalar operator()(const std::vector<Scalar> &x,
                    TinyOpenGL3App *app = nullptr) const {
    system->initialize();
    system->q(0) = Algebra::from_double(M_PI_2);

    policy.set_parameters(x);

    std::vector<Scalar> policy_input(dof), policy_output;

    Scalar cost = Algebra::zero();

    for (int t = 0; t < timesteps; ++t) {
      for (int i = 0; i < dof; ++i) {
        policy_input[i] = system->q(i);
      }
      policy.compute(policy_input, policy_output);
      // apply policy NN output actions as joint forces
      for (int i = 0; i < dof; ++i) {
        system->tau(i) = policy_output[i];
      }
      tds::forward_dynamics(*system, world.get_gravity());
      // clip velocities and accelerations to avoid NaNs over long rollouts
      for (int i = 0; i < dof; ++i) {
        system->qd(i) = Algebra::min(Algebra::from_double(4), system->qd(i));
        system->qd(i) = Algebra::max(Algebra::from_double(-4), system->qd(i));
        system->qdd(i) = Algebra::min(Algebra::from_double(14), system->qdd(i));
        system->qdd(i) =
            Algebra::max(Algebra::from_double(-14), system->qdd(i));
      }
      tds::integrate_euler(*system, dt);

      // the last sphere should be as high as possible, so cost is negative z
      // translation
      cost += -system->links().back().X_world.translation[2];

      if (app != nullptr) {
        // visualize
        TinyVector3f prev_pos(0, 0, 0);
        TinyVector3f color(0, 0, 1);
        float line_width = 1;
        for (const auto &link : *system) {
          TinyVector3f base_pos(
              Algebra::to_double(link.X_world.translation[0]),
              Algebra::to_double(link.X_world.translation[1]),
              Algebra::to_double(link.X_world.translation[2]));
          app->m_renderer->draw_line(prev_pos, base_pos, color, line_width);
          prev_pos = base_pos;
        }
        app->m_renderer->update_camera(2);
        DrawGridData data;
        data.drawAxis = false;
        data.upAxis = 2;
        app->draw_grid(data);
        app->m_renderer->render_scene();
        app->m_renderer->write_transforms();
        app->swap_buffer();
        std::this_thread::sleep_for(
            std::chrono::duration<double>(Algebra::to_double(dt)));
      }
    }
    return cost;
  }
};

TinyOpenGL3App *visualizer = nullptr;

struct PolicyOptExperiment : public tds::Experiment {
  PolicyOptExperiment() : tds::Experiment("policy_optimization") {}

  /**
   * Visualize the current best policy after each evolution.
   */
  void after_iteration(const std::vector<double> &x) override {
    CostFunctor<tds::EigenAlgebra> cost;
    cost(x, visualizer);
  }
};

int main(int argc, char *argv[]) {
  visualizer = new TinyOpenGL3App("pendulum_example_gui", 1024, 768);
  visualizer->m_renderer->init();
  visualizer->set_up_axis(2);
  visualizer->m_renderer->get_active_camera()->set_camera_distance(4);
  visualizer->m_renderer->get_active_camera()->set_camera_pitch(-30);
  visualizer->m_renderer->get_active_camera()->set_camera_target_position(0, 0,
                                                                          0);

  tds::OptimizationProblem<tds::DIFF_CERES, CostFunctor> problem;
  for (std::size_t i = 0; i < param_dim; ++i) {
    // assign parameter names, bounds, and initial values
    problem[i].name = "parameter_" + std::to_string(i);
    problem[i].minimum = -0.5;
    problem[i].maximum = 0.5;
    problem[i].value = problem[i].random_value();
  }
  PolicyOptExperiment experiment;
  // more settings can be made here for the experiment
  // it will use LBFGS from NLOPT with Pagmo by default
  //   experiment["settings"]["pagmo"]["nlopt"]["solver"] = "slsqp";
  experiment.run(problem);

  return EXIT_SUCCESS;
}