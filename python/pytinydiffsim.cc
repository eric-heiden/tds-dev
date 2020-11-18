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

#include "math/tiny/tiny_double_utils.h"

<<<<<<< HEAD
#include "examples/motion_import.h"
#include "examples/tiny_urdf_parser.h"
#include "fix64_scalar.h"
#include "tiny_double_utils.h"
#include "tiny_matrix3x3.h"
#include "tiny_matrix_x.h"
#include "tiny_mb_constraint_solver_spring.h"
#include "multi_body.hpp
#include "tiny_pose.h"
#include "tiny_quaternion.h"
#include "tiny_raycast.h"
#include "tiny_rigid_body.h"
#include "tiny_urdf_structures.h"
#include "tiny_urdf_to_multi_body.h"
#include "tiny_vector3.h"
#include "world.hpp
=======
>>>>>>> c02b5b90cba08605a0c5e292d1da0a9ee8450a01

typedef double TinyDualScalar;
typedef double MyScalar;
typedef ::TINY::DoubleUtils MyTinyConstants;

#include "math/tiny/tiny_algebra.hpp"


typedef TinyAlgebra<double, MyTinyConstants> MyAlgebra;

#include "pytinydiffsim_includes.h"

using namespace TINY;
using namespace tds;

namespace py = pybind11;



PYBIND11_MODULE(pytinydiffsim, m) {


#include "pytinydiffsim.inl"

}