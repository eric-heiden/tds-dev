#pragma once

#include "../spatial_vector.hpp"
#include "third_party/eigen3/Eigen/Eigen"

#include "eigen_vector3.hpp" 

template <typename Scalar, typename Constants>
struct EigenAlgebra {
  using Index = int;
  using Scalar = Scalar;
  using Vector3 = EigenVector3;
};