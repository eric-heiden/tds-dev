#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


include_directories(. ..)

set(DL ${CMAKE_DL_LIBS})

find_package(Ceres REQUIRED)
find_package(tinyxml2 REQUIRED)
find_package(crossguid REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(HDF5 REQUIRED)
find_package(HighFive REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(cppzmq REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

find_package(Eigen3 REQUIRED)
add_definitions(-DUSE_EIGEN)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(ct_core)
find_package(ct_optcon)

# TODO(dmillard): PR vcpkg to get an examples feature for bullet.
add_definitions(-DBT_USE_DOUBLE_PRECISION)
find_package(Bullet)

# TODO(dmillard): File bug on cpp-base64 vcpkg?
add_library(cpp-base64 ../vcpkg_installed/x64-linux/include/cpp-base64/base64.cpp)

# TODO(dmillard): These sections don't really belong here.
add_library(meshcat INTERFACE)
target_link_libraries(meshcat INTERFACE crossguid cpp-base64 ${ZeroMQ_STATIC_LIBRARY} Threads::Threads)

add_library(matplotlib-cpp INTERFACE)
target_link_libraries(matplotlib-cpp INTERFACE Python3::Python)

subdirs(opengl_window)

add_library(pybullet_visualizer_api STATIC pybullet_visualizer_api.cpp)
target_include_directories(pybullet_visualizer_api PUBLIC ${BULLET_INCLUDE_DIRS})
target_link_libraries(
  pybullet_visualizer_api
  BulletRoboticsGUI
  BulletExampleBrowserLib
  BulletRobotics
  BulletFileLoader
  BulletWorldImporter
  BulletSoftBody
  BulletDynamics
  BulletCollision
  BulletInverseDynamicsUtils
  BulletInverseDynamics
  LinearMath
  OpenGLWindow
  gwen
  BussIK
  Bullet3Common
  Threads::Threads
  ${DL}
)

add_executable(billiard_opt_gui billiard_opt_gui.cpp)
target_link_libraries(billiard_opt_gui pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})
target_compile_definitions(billiard_opt_gui PRIVATE STAN_MATH)

add_executable(cartpole_opt cartpole_opt.cpp)
target_link_libraries(cartpole_opt pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})

# TODO(eric-heiden): Update source
# add_executable(ceres_estimation ceres_estimation.cpp )
# target_link_libraries(ceres_estimation pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})

# TODO(eric-heiden): Update source
# add_executable(ceres_estimation_real ceres_estimation_real.cpp )
# target_compile_definitions(ceres_estimation_real PRIVATE -DUSE_MATPLOTLIB)
# target_link_libraries(ceres_estimation_real pybullet_visualizer_api ${DL} ${CERES_LIBRARIES} matplotlib-cpp)

add_executable(neural_pendulum neural_pendulum.cpp)
target_link_libraries(neural_pendulum pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})

add_executable(neural_discovery neural_discovery.cpp)
target_compile_definitions(neural_discovery PRIVATE -DNEURAL_SIM)
target_link_libraries(neural_discovery pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})

# TODO(eric-heiden): Update source
# add_executable(neural_actuator neural_actuator.cpp )
# target_link_libraries(neural_actuator pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})

add_executable(neural_wtf neural_wtf.cpp)
target_link_libraries(neural_wtf pybullet_visualizer_api matplotlib-cpp ${DL} ${CERES_LIBRARIES})

# TODO(eric-heiden): Update source
# add_executable(neural_joint neural_joint.cpp )
# target_link_libraries(neural_joint pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})

# add_executable(neural_push neural_push.cpp)
# target_compile_features(neural_push PRIVATE cxx_std_17)
# target_link_libraries(neural_push pybullet_visualizer_api ${DL} HighFive matplotlib-cpp ${CERES_LIBRARIES} -lstdc++fs)

add_executable(neural_heightfield neural_heightfield.cpp)
target_link_libraries(neural_heightfield pybullet_visualizer_api matplotlib-cpp ${DL} ${CERES_LIBRARIES})

add_executable(policy_optimization policy_optimization.cpp)
target_link_libraries(policy_optimization pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})

add_executable(contact_discovery contact_discovery.cpp)
target_link_libraries(contact_discovery pybullet_visualizer_api ${DL} ${CERES_LIBRARIES})

add_executable(laikago_example_gui laikago_example_gui.cpp)
target_link_libraries(laikago_example_gui pybullet_visualizer_api)

add_executable(laikago_footstep_ik laikago_footstep_ik.cpp)
target_link_libraries(laikago_footstep_ik pybullet_visualizer_api)

add_executable(pendulum_ik_gui pendulum_ik_gui.cpp)
target_link_libraries(pendulum_ik_gui pybullet_visualizer_api)

add_executable(generic_urdf_collision generic_urdf_collision.cpp)
target_link_libraries(generic_urdf_collision pybullet_visualizer_api)

add_executable(gravity_compensation gravity_compensation.cpp)
target_link_libraries(gravity_compensation pybullet_visualizer_api)

add_executable(laikago_meshcat_example laikago_meshcat_example.cpp)
target_link_libraries(laikago_meshcat_example pybullet_visualizer_api meshcat)

add_executable(experiment_neural_swimmer experiment_neural_swimmer.cpp)
target_link_libraries(experiment_neural_swimmer ${CERES_LIBRARIES} tinyxml2)

add_executable(tiny_urdf_parser_meshcat_example tiny_urdf_parser_meshcat_example.cpp)
target_link_libraries(tiny_urdf_parser_meshcat_example tinyxml2 meshcat)

add_executable(meshcat_zmq_example meshcat_zmq_example.cpp)
target_link_libraries(meshcat_zmq_example meshcat)

add_executable(tiny_urdf_parser_example tiny_urdf_parser_example.cpp)
target_link_libraries(tiny_urdf_parser_example tinyxml2)

add_executable(neural_contact neural_contact.cpp)
target_link_libraries(neural_contact opengl_window ${DL} ${CERES_LIBRARIES} pybullet_visualizer_api tinyxml2 meshcat)

add_executable(world_example world_example.cpp)

add_executable(test_xarm test_xarm.cpp)

add_executable(pendulum_example_gui pendulum_example_gui.cpp utils/tiny_clock.cpp utils/tiny_chrome_trace_util.cpp utils/tiny_logging.cpp)
target_link_libraries(pendulum_example_gui opengl_window)

add_executable(
  opengl_test
  opengl_test.cpp
  utils/tiny_clock.cpp
  utils/tiny_chrome_trace_util.cpp
  utils/tiny_logging.cpp
)
include(CMakePrintHelpers)
cmake_print_variables(TINYOBJLOADER_LIBRARIES)
target_include_directories(opengl_test PRIVATE ${TINYOBJLOADER_INCLUDE_DIRS} ${Stb_INCLUDE_DIR})
# TODO(dmillard): find_package should return absolute path?
target_link_directories(opengl_test PRIVATE ../vcpkg_installed/x64-linux/lib)
target_link_libraries(opengl_test opengl_window ${TINYOBJLOADER_LIBRARIES})

add_executable(
  tiny_urdf_parser_opengl_example
  tiny_urdf_parser_opengl_example.cpp
  utils/tiny_clock.cpp
  utils/tiny_chrome_trace_util.cpp
  utils/tiny_logging.cpp
)
target_include_directories(tiny_urdf_parser_opengl_example PRIVATE ${TINYOBJLOADER_INCLUDE_DIRS} ${Stb_INCLUDE_DIR})
# TODO(dmillard): find_package should return absolute path?
target_link_directories(tiny_urdf_parser_opengl_example PRIVATE ../vcpkg_installed/x64-linux/lib)
target_link_libraries(tiny_urdf_parser_opengl_example tinyxml2 opengl_window ${TINYOBJLOADER_LIBRARIES})

add_executable(
  opengl_imgui_test
  opengl_imgui_test.cpp
  utils/tiny_clock.cpp
  utils/tiny_chrome_trace_util.cpp
  utils/tiny_logging.cpp
)
target_link_libraries(opengl_imgui_test opengl_window imgui::imgui)

# TODO(dmillard): Get CT into vcpkg
if(ct_core_FOUND AND ct_optcon_FOUND AND Eigen3_FOUND)
  # Activate CppAD (with CodeGen) and HPIPM for Control Toolbox
  add_definitions(-DCPPAD -DCPPADCG -DHPIPM)

  include_directories(${ct_core_INCLUDE_DIRS})
  include_directories(${ct_optcon_INCLUDE_DIRS})
  include_directories(${EIGEN3_INCLUDE_DIRS})

  add_executable(ct_cartpole ct_cartpole.cpp)
  target_link_libraries(ct_cartpole pybullet_visualizer_api ${DL} Threads::Threads hpipm blasfeo)

  add_executable(ct_contact_trajopt ct_contact_trajopt.cpp)
  target_link_libraries(
    ct_contact_trajopt
    pybullet_visualizer_api ${DL} Threads::Threads hpipm blasfeo
  )

  add_executable(ct_laikago_tracking ct_laikago_tracking.cpp)
  target_link_libraries(
    ct_laikago_tracking
    pybullet_visualizer_api ${DL} Threads::Threads hpipm blasfeo
  )
endif(ct_core_FOUND AND ct_optcon_FOUND AND Eigen3_FOUND)

if(WIN32)
  target_link_libraries(pendulum_example_gui Opengl32)
  target_link_libraries(tiny_urdf_parser_opengl_example Opengl32)
  target_link_libraries(opengl_test Opengl32)
  target_link_libraries(opengl_imgui_test Opengl32)
endif()
