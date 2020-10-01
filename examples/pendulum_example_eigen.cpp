// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fenv.h>
#include <stdio.h>

#include <chrono>  // std::chrono::seconds
#include <thread>  // std::this_thread::sleep_for

// #define DEBUG 1

#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/kinematics.hpp"
#include "utils/conversion.hpp"
#include "math/eigen_algebra.hpp"
#include "math/tiny/fix64_scalar.h"
#include "math/tiny/tiny_double_utils.h"
#include "multi_body.hpp"
#include "visualizer/opengl/visualizer.h"
#include "utils/pendulum.hpp"
#include "utils/file_utils.hpp"
#include "world.hpp"

int main(int argc, char* argv[]) {
  //typedef TinyAlgebra<double, DoubleUtils> Algebra;
  typedef tds::EigenAlgebra Algebra;

  typedef typename Algebra::Vector3 Vector3;
  typedef typename Algebra::Quaternion Quaternion;
  typedef typename Algebra::VectorX VectorX;
  typedef typename Algebra::Matrix3 Matrix3;
  typedef typename Algebra::Matrix3X Matrix3X;
  typedef typename Algebra::MatrixX MatrixX;
  typedef tds::RigidBody<Algebra> RigidBody;
  typedef tds::RigidBodyContactPoint<Algebra> RigidBodyContactPoint;
  typedef tds::MultiBody<Algebra> MultiBody;
  typedef tds::MultiBodyContactPoint<Algebra> MultiBodyContactPoint;
  typedef tds::Transform<Algebra> Transform;

  TinyOpenGL3App app("pendulum_example_gui", 1024, 768);
  app.m_renderer->init();
  app.set_up_axis(2);
  app.m_renderer->get_active_camera()->set_camera_distance(4);
  app.m_renderer->get_active_camera()->set_camera_pitch(-30);
  app.m_renderer->get_active_camera()->set_camera_target_position(0, 0, 0);
  // install ffmpeg in path and uncomment, to enable video recording
  app.dump_frames_to_video("test.mp4");

  // Set NaN trap
  tds::activate_nan_trap();

  tds::World<Algebra> world;

  std::vector<RigidBody*> bodies;
  std::vector<int> visuals;

  std::vector<MultiBody*> mbbodies;
  std::vector<int> mbvisuals;

  int num_spheres = 5;

  MultiBody* mb = world.create_multi_body();
  init_compound_pendulum<Algebra>(*mb, world, num_spheres);

  mbbodies.push_back(mb);

  int sphere_shape = app.register_graphics_unit_sphere_shape(SPHERE_LOD_HIGH);

  for (int i = 0; i < num_spheres; i++) {
    TinyVector3f pos(0, i * 0.1, 0);
    TinyQuaternionf orn(0, 0, 0, 1);
    TinyVector3f color(0.6, 0.6, 1);
    TinyVector3f scaling(0.05, 0.05, 0.05);
    int instance = app.m_renderer->register_graphics_instance(
        sphere_shape, pos, orn, color, scaling);
    mbvisuals.push_back(instance);
  }

#if 0
  if (visualizer->canSubmitCommand()) {
    for (int i = 0; i < mb->m_links.size(); i++) {
      int sphereId = visualizer->loadURDF("sphere_small.urdf");
      mbvisuals.push_back(sphereId);
      // apply some linear joint damping
      mb->m_links[i].m_damping = 5.;
    }
  }
#endif
  // mb->q() = std::vector<double>(mb->dof(), DoubleUtils::zero());
  // mb->qd() = std::vector<double>(mb->dof_qd(), DoubleUtils::zero());
  // mb->tau() = std::vector<double>(mb->dof_qd(), DoubleUtils::zero());
  // mb->qdd() = std::vector<double>(mb->dof_qd(), DoubleUtils::zero());

  mb->q() = Algebra::zerox(mb->dof());
  mb->qd() = Algebra::zerox(mb->dof_qd());
  mb->tau() = Algebra::zerox(mb->dof_qd());
  mb->qdd() = Algebra::zerox(mb->dof_qd());

  Vector3 gravity(0., 0., -9.81);

  MatrixX M(mb->links().size(), mb->links().size());

  double dt = 1. / 240.;
  app.set_mp4_fps(1. / dt);
  int upAxis = 2;
  // while (!app.m_window->requested_exit()) {
  for (int step = 0; step < 1000; ++step) {
    app.m_renderer->update_camera(upAxis);
    DrawGridData data;
    data.upAxis = upAxis;
    app.draw_grid(data);

    tds::forward_kinematics(*mb);

    { world.step(dt); }

    {
      tds::forward_dynamics(*mb, gravity);
      mb->clear_forces();
    }

    {
      tds::integrate_euler(*mb, mb->q(), mb->qd(), mb->qdd(), dt);

      tds::mass_matrix(*mb, &M);
      // Algebra::print("Mass matrix", M);

      mb->print_state();

      if (mb->qd()[0] < -1e4) {
        assert(0);
      }
    }

    std::this_thread::sleep_for(std::chrono::duration<double>(dt));
    // sync transforms
    int visual_index = 0;
    TinyVector3f prev_pos(0, 0, 0);
    TinyVector3f color(0, 0, 1);
    float line_width = 1;

    typedef tds::Conversion<Algebra, tds::TinyAlgebraf> Conversion;

    if (!mbvisuals.empty()) {
      for (int b = 0; b < mbbodies.size(); b++) {
        for (int l = 0; l < mbbodies[b]->links().size(); l++) {
          const MultiBody* body = mbbodies[b];
          if (body->links()[l].X_visuals.empty()) continue;

          int sphereId = mbvisuals[visual_index++];

          const Transform& geom_X_world =
              body->links()[l].X_world * body->links()[l].X_visuals[0];
          TinyVector3f base_pos = Conversion::convert(geom_X_world.translation);
          TinyQuaternionf base_orn = Conversion::convert(
              Algebra::matrix_to_quat(geom_X_world.rotation));
          if (l >= 0) {
            app.m_renderer->draw_line(prev_pos, base_pos, color, line_width);
          }
          prev_pos = base_pos;
          app.m_renderer->write_single_instance_transform_to_cpu(
              base_pos, base_orn, sphereId);
        }
      }
    }
    app.m_renderer->render_scene();
    app.m_renderer->write_transforms();
    app.swap_buffer();
  }

  return 0;
}
