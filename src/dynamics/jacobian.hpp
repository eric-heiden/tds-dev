#pragma once

#include "kinematics.hpp"

namespace tds {

/**
 * Computes the body Jacobian for a point in world frame on certain link.
 * This function does not update the robot configuration with the given
 * joint positions.
 */
template <typename Algebra>
typename Algebra::Matrix3X point_jacobian(
    MultiBody<Algebra> &mb, const typename Algebra::VectorX &q, int link_index,
    const typename Algebra::Vector3 &world_point) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix3X = typename Algebra::Matrix3X;
  typedef tds::Transform<Algebra> Transform;
  typedef tds::MotionVector<Algebra> MotionVector;
  typedef tds::ForceVector<Algebra> ForceVector;
  typedef tds::Link<Algebra> Link;

  assert(Algebra::size(q) == mb.dof());
  assert(link_index < static_cast<int>(mb.size()));
  Matrix3X jac(3, mb.dof_qd());
  jac.set_zero();
  std::vector<Transform> links_X_world;
  std::vector<Transform> links_X_base;
  Transform base_X_world;
  forward_kinematics_q(mb, q, &base_X_world, &links_X_world, &links_X_base);
  Transform point_tf;
  point_tf.set_identity();
  point_tf.translation = world_point;
  if (mb.is_floating()) {
    // see (Eq. 2.238) in
    // https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/FloatingBaseKinematics.pdf
    Vector3 base_to_point = world_point - base_X_world.translation;
    Matrix3 cr = Algebra::cross_matrix(base_to_point);
    jac[0] = cr[0];
    jac[1] = cr[1];
    jac[2] = cr[2];
    jac[3][0] = Algebra::one();
    jac[4][1] = Algebra::one();
    jac[5][2] = Algebra::one();
  } else {
    point_tf.translation = world_point;
  }
  // loop over all links that lie on the path from the given link to world
  if (link_index >= 0) {
    Link *body = &mb[link_index];
    while (true) {
      int i = body->index;
      if (body->joint_type != JOINT_FIXED) {
        MotionVector st = links_X_world[i].apply_inverse(body->S);
        MotionVector xs = point_tf.apply(st);
        jac[body->qd_index] = xs.bottom;
      }
      if (body->parent_index < 0) break;
      body = &mb[body->parent_index];
    }
  }
  return jac;
}

template <typename Algebra>
typename Algebra::Matrix3X point_jacobian(
    MultiBody<Algebra> &mb, int link_index,
    const typename Algebra::Vector3 &world_point) {
  return point_jacobian(mb, mb.q(), link_index, world_point);
}

/**
 * Estimate the point Jacobian using finite differences.
 * This function should only be called for testing purposes.
 */
template <typename Algebra>
typename Algebra::Matrix3X point_jacobian_fd(
    const MultiBody<Algebra> &mb, const typename Algebra::VectorX &q,
    int link_index, const typename Algebra::Vector3 &start_point,
    const typename Algebra::Scalar &eps = Algebra::fraction(1, 10000)) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using VectorX = typename Algebra::VectorX;
  using Matrix3X = typename Algebra::Matrix3X;
  using Quaternion = typename Algebra::Quaternion;
  typedef tds::Transform<Algebra> Transform;
  typedef tds::MotionVector<Algebra> MotionVector;
  typedef tds::ForceVector<Algebra> ForceVector;
  typedef tds::Link<Algebra> Link;

  assert(Algebra::size(q) == mb.dof());
  assert(link_index < static_cast<int>(mb.size()));
  Matrix3X jac(3, mb.dof_qd());
  jac.set_zero();
  std::vector<Transform> links_X_world;
  Transform base_X_world;
  // compute world point transform for the initial joint angles
  forward_kinematics_q(q, &base_X_world, &links_X_world);
  if (mb.empty()) return jac;
  // convert start point in world coordinates to link frame
  const Vector3 base_point =
      links_X_world[link_index].apply_inverse(start_point);
  Vector3 world_point;

  VectorX q_x;
  Transform base_X_world_temp;
  for (int i = 0; i < mb.dof_qd(); ++i) {
    q_x = q;
    if (mb.is_floating() && i < 3) {
      // special handling of quaternion differencing via angular velocity
      Quaternion base_rot = Algebra::matrix_to_quat(base_X_world.rotation);

      Vector3 angular_velocity;
      angular_velocity.set_zero();
      angular_velocity[i] = Algebra::one();

      base_rot += (angular_velocity * base_rot) * (eps * Algebra::half());
      base_rot.normalize();
      q_x[0] = base_rot.getX();
      q_x[1] = base_rot.getY();
      q_x[2] = base_rot.getZ();
      q_x[3] = base_rot.getW();
    } else {
      // adjust for the +1 offset with the 4 DOF orientation in q vs. 3 in qd
      int q_index = mb.is_floating() ? i + 1 : i;
      q_x[q_index] += eps;
    }
    forward_kinematics_q(mb, q_x, &base_X_world_temp, &links_X_world);
    world_point = links_X_world[link_index].apply(base_point);
    for (int j = 0; j < 3; ++j)
      jac(j, i) = (world_point[j] - start_point[j]) / eps;
  }

  return jac;
}

}  // namespace tds