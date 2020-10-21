#pragma once

#include <cppad/cg.hpp>

#include "base.hpp"

namespace tds {
template <typename Scalar>
struct is_cppad_scalar {
  static constexpr bool value = false;
};
template <typename Scalar>
struct is_cppad_scalar<CppAD::AD<Scalar>> {
  static constexpr bool value = true;
};
template <typename Scalar>
struct is_cppad_scalar<CppAD::cg::CG<Scalar>> {
  static constexpr bool value = true;
};

template <typename Scalar>
static TINY_INLINE CppAD::AD<Scalar> where_gt(
    const CppAD::AD<Scalar>& x, const CppAD::AD<Scalar>& y,
    const CppAD::AD<Scalar>& if_true, const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpGt(x, y, if_true, if_false);
}
template <typename Scalar>
static TINY_INLINE Scalar where_gt(const Scalar& x, const Scalar& y,
                                   const Scalar& if_true,
                                   const Scalar& if_false) {
  return x > y ? if_true : if_false;
}

template <typename Scalar>
static TINY_INLINE CppAD::AD<Scalar> where_ge(
    const CppAD::AD<Scalar>& x, const CppAD::AD<Scalar>& y,
    const CppAD::AD<Scalar>& if_true, const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpGe(x, y, if_true, if_false);
}
template <typename Scalar>
static TINY_INLINE Scalar where_ge(const Scalar& x, const Scalar& y,
                                   const Scalar& if_true,
                                   const Scalar& if_false) {
  return x >= y ? if_true : if_false;
}

template <typename Scalar>
static TINY_INLINE CppAD::AD<Scalar> where_lt(
    const CppAD::AD<Scalar>& x, const CppAD::AD<Scalar>& y,
    const CppAD::AD<Scalar>& if_true, const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpLt(x, y, if_true, if_false);
}
template <typename Scalar>
static TINY_INLINE Scalar where_lt(const Scalar& x, const Scalar& y,
                                   const Scalar& if_true,
                                   const Scalar& if_false) {
  return x < y ? if_true : if_false;
}

template <typename Scalar>
static TINY_INLINE CppAD::AD<Scalar> where_le(
    const CppAD::AD<Scalar>& x, const CppAD::AD<Scalar>& y,
    const CppAD::AD<Scalar>& if_true, const CppAD::AD<Scalar>& if_false) {
  return CppAD::CondExpLe(x, y, if_true, if_false);
}
template <typename Scalar>
static TINY_INLINE Scalar where_le(const Scalar& x, const Scalar& y,
                                   const Scalar& if_true,
                                   const Scalar& if_false) {
  return x <= y ? if_true : if_false;
}
}  // namespace tds