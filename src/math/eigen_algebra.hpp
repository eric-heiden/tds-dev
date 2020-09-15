#pragma once


#include <iostream>

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
  using Matrix3X = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
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
    // This constructor exist: Quaternion(w, x, y, z);
    // The order is different from Enoki
    quat = Quaternion(1., 0., 0., 0.);
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

  template <int Size1, int Size2>
  EIGEN_ALWAYS_INLINE static void assign_block(
      Eigen::Matrix<Scalar, Size1, Size1> &output,
      const Eigen::Matrix<Scalar, Size2, Size2> &input, int i, int j,
      int m = Size2, int n = Size2, int input_i = 0,
      int input_j = 0) {
    assert(i + m <= Size1 && j + n <= Size1);
    assert(input_i + m <= Size2 && input_j + n <= Size2);
    for (int ii = 0; ii < m; ++ii) {
      for (int jj = 0; jj < n; ++jj) {
        output(ii + i, jj + j) = input(ii + input_i, jj + input_j);
      }
    }
  }

  template <int Size>
  EIGEN_ALWAYS_INLINE static void assign_column(Eigen::Matrix<Scalar, Size, Size> &m,
                                         Index i,
                                         const Eigen::Array<Scalar, Size, 1> &v) {
    m.col(i) = v;
  }

  EIGEN_ALWAYS_INLINE static Matrix3 quat_to_matrix(const Quaternion &quat) {
    // NOTE: Eigen requires quat to be normalized
    return quat.toRotationMatrix();
  }
  EIGEN_ALWAYS_INLINE static Matrix3 quat_to_matrix(const Scalar &x, const Scalar &y,
                                             const Scalar &z, const Scalar &w) {
    return Quaternion(w, x, y, z).toRotationMatrix();
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

  EIGEN_ALWAYS_INLINE static Matrix3 rotation_x_matrix(const Scalar &angle) {
    Scalar c = std::cos(angle);
    Scalar s = std::sin(angle);
    Matrix3 temp;
    temp << 1, 0, 0, 
            0, c, s,
            0, -s, c;
    return temp;
  }

  EIGEN_ALWAYS_INLINE static Matrix3 rotation_y_matrix(const Scalar &angle) {
    Scalar c = std::cos(angle);
    Scalar s = std::sin(angle);
    Matrix3 temp;
    temp << c, 0, -s, 
            0, 1, 0,
            s, 0, c;
    return temp;
  }

  EIGEN_ALWAYS_INLINE static Matrix3 rotation_z_matrix(const Scalar &angle) {
    Scalar c = std::cos(angle);
    Scalar s = std::sin(angle);
    Matrix3 temp;
    temp << c, s, 0, 
            -s, c, 0,
            0, 0, 1;
    return temp;
  }

  static Matrix3 rotation_zyx_matrix(const Scalar &r, const Scalar &p,
                                     const Scalar &y) {
    Scalar ci(std::cos(r));
    Scalar cj(std::cos(p));
    Scalar ch(std::cos(y));
    Scalar si(std::sin(r));
    Scalar sj(std::sin(p));
    Scalar sh(std::sin(y));
    Scalar cc = ci * ch;
    Scalar cs = ci * sh;
    Scalar sc = si * ch;
    Scalar ss = si * sh;
    return Matrix3(cj * ch, sj * sc - cs, sj * cc + ss, cj * sh, sj * ss + cc,
                   sj * cs - sc, -sj, cj * si, cj * ci);

    Matrix3 tmp;
    tmp << cj * ch, sj * sc - cs, sj * cc + ss,
           cj * sh, sj * ss + cc, sj * cs - sc,
           -sj,     cj * si,      cj * ci;
    return tmp;
  }

  EIGEN_ALWAYS_INLINE static Vector3 rotate(const Quaternion &q, const Vector3 &v) {
    // NOTE: Eigen supports direct concatenation of quaternions and vector
    return q * v;
  }

  /**
   * Computes the quaternion delta given current rotation q, angular velocity w,
   * time step dt.
   */
  EIGEN_ALWAYS_INLINE static Quaternion quat_velocity(const Quaternion &q,
                                               const Vector3 &w,
                                               const Scalar &dt) {
    Quaternion delta((-q[0] * w[0] - q[1] * w[1] - q[2] * w[2]) * (0.5 * dt),
                     (q[3] * w[0] + q[1] * w[2] - q[2] * w[1]) * (0.5 * dt),
                     (q[3] * w[1] + q[2] * w[0] - q[0] * w[2]) * (0.5 * dt),
                     (q[3] * w[2] + q[0] * w[1] - q[1] * w[0]) * (0.5 * dt)
                    );
    // Broadcast not supported
    // delta *= 0.5 * dt;
    return delta;
  }

  EIGEN_ALWAYS_INLINE static const Scalar &quat_x(const Quaternion &q) { return q.x(); }
  EIGEN_ALWAYS_INLINE static const Scalar &quat_y(const Quaternion &q) { return q.y(); }
  EIGEN_ALWAYS_INLINE static const Scalar &quat_z(const Quaternion &q) { return q.z(); }
  EIGEN_ALWAYS_INLINE static const Scalar &quat_w(const Quaternion &q) { return q.w(); }
  EIGEN_ALWAYS_INLINE static const Quaternion quat_from_xyzw(const Scalar& x,
                                                             const Scalar& y,
                                                             const Scalar& z,
                                                             const Scalar& w)
  {
    // Eigen specific constructor coefficient order
    return Quaternion(w, x, y, z);
  }

  template <int Size1, int Size2>
  EIGEN_ALWAYS_INLINE static void set_zero(Eigen::Matrix<Scalar, Size1, Size2> &m) {
    m.setConstant(m);
  }
  template <int Size1, int Size2 = 1>
  EIGEN_ALWAYS_INLINE static void set_zero(Eigen::Array<Scalar, Size1, Size2> &v) {
    v.setZero();
  }
  EIGEN_ALWAYS_INLINE static void set_zero(MotionVector &v) {
    v.top.setZero();
    v.bottom.setZero();
  }
  EIGEN_ALWAYS_INLINE static void set_zero(ForceVector &v) {
    v.top.setZero();
    v.bottom.setZero();
  }

  /**
   * Non-differentiable comparison operator.
   */
  EIGEN_ALWAYS_INLINE static bool less_than(const Scalar &a, const Scalar &b) {
    return a < b;
  }

  /**
   * Non-differentiable comparison operator.
   */
  EIGEN_ALWAYS_INLINE static bool less_than_zero(const Scalar &a) {
    return a < 0.;
  }

  /**
   * Non-differentiable comparison operator.
   */
  EIGEN_ALWAYS_INLINE static bool greater_than_zero(const Scalar &a) {
    return a > 0.;
  }

  /**
   * Non-differentiable comparison operator.
   */
  EIGEN_ALWAYS_INLINE static bool greater_than(const Scalar &a, const Scalar &b) {
    return a > b;
  }

  /**
   * Non-differentiable comparison operator.
   */
  EIGEN_ALWAYS_INLINE static bool equals(const Scalar &a, const Scalar &b) {
    return a == b;
  }

  TINY_INLINE static double to_double(const Scalar& s) {
    return static_cast<double>(s);
  }

  TINY_INLINE static Scalar from_double(double s) {
    return static_cast<Scalar>(s);
  }


  template <int Size1, int Size2>
  static void print(const std::string &title, Eigen::Matrix<Scalar, Size1, Size2> &m) {
    std::cout << title << "\n" << m << std::endl;
  }
  template <int Size1, int Size2 = 1>
  static void print(const std::string &title, Eigen::Array<Scalar, Size1, Size2> &v) {
    std::cout << title << "\n" << v << std::endl;
  }
  static void print(const std::string &title, const Scalar &v) {
    std::cout << title << "\n" << to_double(v) << std::endl;
  }
  template <typename T>
  static void print(const std::string &title, const T &abi) {
    abi.print(title.c_str());
  }

  template <typename T>
  TINY_INLINE static auto sin(const T &s) {
    return std::sin(s);
  }

  template <typename T>
  TINY_INLINE static auto cos(const T &s) {
    return std::cos(s);
  }

  template <typename T>
  TINY_INLINE static auto abs(const T &s) {
    return std::abs(s);
  }

  EigenAlgebraT<Scalar>() = delete;
};

typedef EigenAlgebraT<double> EigenAlgebra;

} // end namespace tds