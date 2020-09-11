#pragma once

#include "third_party/eigen3/Eigen/Core"

#include "spatial_vector.hpp"

namespace tds
{

template <typename ScalarT = double>
struct EigenAlgebraT 
{
  using Index = Eigen::Index;
  using Scalar = ScalarT;
  using EigenAlgebra = EigenAlgebraT<Scalar>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
  using Matrix6 = Eigen::Matrix<Scalar, 6, 6>;
  using Quaternion = Eigen::Quaternion<Scalar>;
  using SpatialVector = tds::SpatialVector<EigenAlgebra>;
  using MotionVector = tds::MotionVector<EigenAlgebra>;
  using ForceVector = tds::ForceVector<EigenAlgebra>;

  template <typename T>
  EIGEN_ALWAYS_INLINE static auto transpose(const T& matrix) {
    return matrix.transpose();
  }

  template <typename T> 
  EIGEN_ALWAYS_INLINE static auto inverse(const T& matrix) {
    return matrix.inverse();
  }

  template <typename T>
  EIGEN_ALWAYS_INLINE static auto inverse_transpose(const T& matrix) {
    return matrix.inverse().transpose();
  }

  template <typename T1, typename T2>
  EIGEN_ALWAYS_INLINE static auto cross(const T1& vector_a, const T2& vector_b) {
    return vector_a.cross(vector_b);
  }

  /**
   * V1 = mv(w1, v1)
   * V2 = mv(w2, v2)
   * V1 x V2 = mv(w1 x w2, w1 x v2 + v1 x w2)
   */
  static inline MotionVector cross(const MotionVector &a,
                                   const MotionVector &b) {
    return MotionVector(
      a.top.cross(b.top),
      a.top.cross(b.top) + a.bottom.cross(b.bottom)
    );
  }

  /**
   * V = mv(w, v)
   * F = fv(n, f)
   * V x* F = fv(w x n + v x f, w x f)
   */
  static inline ForceVector cross(const MotionVector &a, const ForceVector &b) {
    return ForceVector(
      a.top.cross(b.top) + a.bottom.cross(b.bottom),
      a.top.cross(b.top)
    );
  }

  EIGEN_ALWAYS_INLINE static Index size(const VectorX& v) {
    return v.size();
  }

  /**
   * V = mv(w, v)
   * F = mv(n, f)
   * V.F = w.n + v.f
   */
  EIGEN_ALWAYS_INLINE static Scalar dot(const MotionVector &a, const ForceVector &b) {
    return a.top.dot(b.top) + a.bottom.dot(b.bottom);
    // return enoki::dot(a.top, b.top) + enoki::dot(a.bottom, b.bottom);
  }
  EIGEN_ALWAYS_INLINE static Scalar dot(const ForceVector &a, const MotionVector &b) {
    return dot(b, a);
  }

  template <typename T1, typename T2>
  EIGEN_ALWAYS_INLINE static auto dot(const T1& vector_a, const T2& vector_b) {
    return vector_a.dot(vector_b);
  }

  // Eigen sqrt defaults to std::sqrt. Eigen::sqrt is a broadcast operation
  TINY_INLINE static Scalar norm(const MotionVector &v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] +
                       v[4] * v[4] + v[5] * v[5]);
  }

  TINY_INLINE static Scalar norm(const ForceVector &v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] +
                       v[4] * v[4] + v[5] * v[5]);
  }
  template <typename T>
  EIGEN_ALWAYS_INLINE static Scalar norm(const T &v) {
    return v.norm();
  }
  template <typename T>
  EIGEN_ALWAYS_INLINE static Scalar sqnorm(const T &v) {
    return v.squaredNorm();
  }

  template <typename T>
  EIGEN_ALWAYS_INLINE static auto normalize(T &v) {
    v.normalize();
    return v;
  }

  EIGEN_ALWAYS_INLINE static Matrix3 cross_matrix(const Vector3 &v) {
    Matrix3 tmp;
    tmp << 0., -v(2, 0), v(1,0), 
           v(2,0), 0., -v(0,0), 
           -v(1,0), v(0,0), 0.;
    return tmp;
  }

  EIGEN_ALWAYS_INLINE static Matrix3 zero33() { 
    return Matrix3::Zero();
  }

  EIGEN_ALWAYS_INLINE static VectorX zerox(Index size) {
    return VectorX::Zero(size);
  }

  EIGEN_ALWAYS_INLINE static Matrix3 diagonal3(const Vector3 &v) {
    return Matrix3(v[0], 0, 0, 0, v[1], 0, 0, 0, v[2]);
    Matrix3 tmp;
    tmp << v(0,0), 0, 0,
           0, v(1,0), 0,
           0, 0, v(2,0);
    return tmp;
  }

  EIGEN_ALWAYS_INLINE static Matrix3 diagonal3(const Scalar &v) { 
    Matrix3 tmp;
    tmp << v, 0, 0,
           0, v, 0,
           0, 0, v;
    return tmp;
  }

  EIGEN_ALWAYS_INLINE static Matrix3 eye3() { 
    return Matrix3::Identity();
  }
  EIGEN_ALWAYS_INLINE static void set_identity(Quaternion &quat) {
    // This constructor exist: Quaternion(x, y, z, w);
    quat = Quaternion(0., 0., 0., 1.);
  }

  EIGEN_ALWAYS_INLINE static Scalar zero() { return 0; }
  EIGEN_ALWAYS_INLINE static Scalar one() { return 1; }
  EIGEN_ALWAYS_INLINE static Scalar two() { return 2; }
  EIGEN_ALWAYS_INLINE static Scalar half() { return 0.5; }
  EIGEN_ALWAYS_INLINE static Scalar pi() { return M_PI; }
  EIGEN_ALWAYS_INLINE static Scalar fraction(int a, int b) { return ((double)a) / b; }

  static Scalar scalar_from_string(const std::string &s) {
    return std::stod(s);
  }

  EIGEN_ALWAYS_INLINE static Vector3 zero3() { return Vector3(0); }
  EIGEN_ALWAYS_INLINE static Vector3 unit3_x() { return Vector3(1, 0, 0); }
  EIGEN_ALWAYS_INLINE static Vector3 unit3_y() { return Vector3(0, 1, 0); }
  EIGEN_ALWAYS_INLINE static Vector3 unit3_z() { return Vector3(0, 0, 1); }

  template <std::size_t Size1, std::size_t Size2>
  EIGEN_ALWAYS_INLINE static void assign_block(
      Eigen::Matrix<Scalar, Size1, Size1> &output,
      const Eigen::Matrix<Scalar, Size2, Size2> &input, std::size_t i, std::size_t j,
      std::size_t m = Size2, std::size_t n = Size2, std::size_t input_i = 0,
      std::size_t input_j = 0) {
    assert(i + m <= Size1 && j + n <= Size1);
    assert(input_i + m <= Size2 && input_j + n <= Size2);
    for (std::size_t ii = 0; ii < m; ++ii) {
      for (std::size_t jj = 0; jj < n; ++jj) {
        output(ii + i, jj + j) = input(ii + input_i, jj + input_j);
      }
    }
  }

  template <std::size_t Size>
  EIGEN_ALWAYS_INLINE static void assign_column(Eigen::Matrix<Scalar, Size, Size> &m,
                                         std::size_t i,
                                         const Eigen::Array<Scalar, Size, 1> &v) {
    m.col(i) = v;
  }

  EIGEN_ALWAYS_INLINE static Matrix3 quat_to_matrix(const Quaternion &quat) {
    // NOTE: Eigen requires quat to be normalized
    return quat.toRotationMatrix();
  }
  EIGEN_ALWAYS_INLINE static Matrix3 quat_to_matrix(const Scalar &x, const Scalar &y,
                                             const Scalar &z, const Scalar &w) {
    return Quaternion(x, y, z, w).toRotationMatrix();
  }
  EIGEN_ALWAYS_INLINE static Quaternion matrix_to_quat(const Matrix3 &m) {
    return Quaternion(m);
  }
  EIGEN_ALWAYS_INLINE static Quaternion axis_angle_quaternion(const Vector3 &axis,
                                                       const Scalar &angle) {
    // return enoki::rotate<Quaternion, Vector3>(axis, angle);
    // TODO: Check if this is equivalent in Eigen
    return Quaternion(Eigen::AngleAxis(angle, axis));
  }

  // TODO: Continue from rotation_x_matrix

};

} // end namespace tds