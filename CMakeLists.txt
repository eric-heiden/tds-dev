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

# Tiny Differentiable Simulator or TDS

cmake_minimum_required(VERSION 2.4.3)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON) # Relative RPATH make AppImage easier.
set(CMAKE_CXX_STANDARD 17)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW) # Absolute path libs
  cmake_policy(SET CMP0011 NEW) # Automatic push and pop
  cmake_policy(SET CMP0017 NEW) # Prefer files from CMake module dir
  cmake_policy(SET CMP0028 NEW) # Double colon libraries
  cmake_policy(SET CMP0042 NEW) # Enable MACOXS_RPATH
endif(COMMAND cmake_policy)

# This line has to appear before 'PROJECT' in order to be able to disable
# incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(DIFF_PHYSICS)

# The core library consists of these headers
# TODO(dmillard): Is this necessary?
file(GLOB TDS_HDRS "*.h")

if(WIN32)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Shared Libs" FORCE)
else(WIN32)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "Shared Libs" FORCE)
endif(WIN32)

# TODO(dmillard): Is this necessary?
add_definitions(-DSTATIC_LINK_SPD_PLUGIN)

subdirs(examples)

option(BUILD_PYTHON_PLUGIN "Build python bindings?" OFF)
if(BUILD_PYTHON_PLUGIN)
  subdirs(python)
endif()
