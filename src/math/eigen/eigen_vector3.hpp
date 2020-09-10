#pragma once

#include "third_party/eigen3/Eigen/Eigen"

template <typename Scalar, typename Constants>
struct EigenVector3 {
  typedef Eigen::Matrix<Scalar, 3, 1> EigenVector3Impl;

  EigenVector3Impl m_v;

  int m_size{3};

  explicit EigenVector3(int unused = 0) {}

  EigenVector3(const EigenVector3& rhs)
  : m_v(rhs.m_v) 
  {}

  EigenVector3(const EigenVector3& rhs) {
    m_v = rhs.m_v;
  }

  EigenVector3(const EigenVector3Impl& ev) 
  : m_v(ev)
  {}

  EigenVector3(Scalar x, Scalar y, Scalar z) {
    m_v << x, y, z;
  }

  Scalar x() const { return m_v(0, 0); }
  Scalar getX() const { return m_v(0, 0); }
  void setX(Scalar x) { m_v(0, 0) = x; }
  
  Scalar y() const { return m_v(1, 0); }
  Scalar getY() const { return m_v(1, 0); }
  void setY(Scalar y) { m_v(1, 0) = y; }

  Scalar z() const { return m_v(2, 0); }
  Scalar getZ() const { return m_v(2, 0); }
  void setZ(Scalar z) { m_v(2, 0) = z; }

  void setValue(const Scalar& x, const Scalar& y, const Scalar& z) {
    m_v << x, y, z;
  }

  void set_zero() {
    setValue(Constants::zero(), Constants::zero(), Constants::zero());
  }

  static EigenVector3 zero() {
    EigenVector3 res(Constants::zero(), Constants::zero(), Constants::zero());

    return res;
  }

  static EigenVector3 makeUnitX() {
    EigenVector3 res(Constants::one(), Constants::zero(), Constants::zero());

    return res;
  }

  static EigenVector3 makeUnitY() {
    EigenVector3 res(Constants::zero(), Constants::one(), Constants::zero());

    return res;
  }

  static EigenVector3 makeUnitZ() {
    EigenVector3 res(Constants::zero(), Constants::zero(), Constants::one());

    return res;
  }

  static EigenVector3 create(const Scalar& x, const Scalar& y, const Scalar& z) {
    EigenVector3 res(x, y, z);
    return res;
  }

  inline Scalar dot(const EigenVector3& other) const {
    Scalar res = m_v.dot(other);
    return res;
  }

  static Scalar dot2(const EigenVector3& a, const EigenVector3& b) {
    Scalar res = a.dot(b);
    return res;
  }

  inline EigenVector3 cross(const EigenVector3& v) const {
    auto res_v = m_v.cross(v.m_v);
    EigenVector3 res(res_v);
    return res;
  }

  static EigenVector3 cross2(const EigenVector3& a, const EigenVector3& b) {
    EigenVector3 res = a.cross(b);
    return res;
  }

  inline Scalar length() const {
    Scalar res = this->dot(*this);
    if (res == Constatns::zero()) return res;
    res = Constants::sqrt1(res);
    return res;
  }

  inline Scalar length_squared() const {
    Scalar res = this->dot(*this);
    return res;
  }

  inline void normalize() {
    m_v.normalize();
  }

  inline EigenVector3 normalized() const {
    EigenVector3 tmp(*this);
    tmp.normalize();
    return tmp;
  }

  inline Scalar sqnorm() const {
    Scalar res = m_v.squaredNorm();
    return res;
  }

  inline EigenVector3& operator+=(const EigenVector3& v) {
    m_v += v.m_v;
    return *this;
  }

  inline EigenVector3& operator-=(const EigenVector3& v) {
    m_v -= v.m_v;
    return *this;
  }

  inline EigenVector3& operator*=(const EigenVector3& v) {
    // Coefficient multiplication must be done with Arrays
    // Aligning with behavior from TinyVector
    Eigen::Array<Scalar, 3, 1> tmp = m_v.array();
    tmp *= v.m_v.array();
    m_v = tmp.matrix();
  }

  inline EigenVector3 operator-() const {
    EigenVector3 v = EigenVector3::Create(-getX(), -getY(), -getZ());
    return v;
  }

  inline Scalar& operator[](int i) {
    assert(i < 3 && i >= 0);
    return m_v(i, 0);
  }

  inline const Scalar& operator[](int i) const {
    assert(i < 3 && i >= 0);
    return m_v(i, 0);
  }

};