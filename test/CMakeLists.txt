enable_testing()
    
find_package(GTest )
if (GTest_FOUND)

find_package(Ceres )

if(Ceres_FOUND)
  add_executable(test_gradients test_gradients.cpp)
  target_link_libraries(test_gradients PRIVATE ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${CERES_LIBRARIES})
  add_test(GradientTests test_gradients)

  add_executable(test_ceres_estimation test_ceres_estimation.cpp)
  target_link_libraries(test_ceres_estimation PRIVATE ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${CERES_LIBRARIES})
  add_test(EstimationTests test_ceres_estimation)

  add_executable(test_pagmo_estimation test_pagmo_estimation.cpp)
  target_link_libraries(test_pagmo_estimation PRIVATE ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${CERES_LIBRARIES} ${CMAKE_DL_LIBS})
  add_test(EstimationTests test_pagmo_estimation)
endif(Ceres_FOUND)

if (USE_STAN)
    target_link_libraries(test_gradients PRIVATE ${STAN_LIBRARIES})
    if(Ceres_FOUND)
      target_link_libraries(test_ceres_estimation PRIVATE ${STAN_LIBRARIES})
    endif(Ceres_FOUND)
    target_link_libraries(test_pagmo_estimation PRIVATE ${STAN_LIBRARIES})

    # target_compile_definitions(test_gradients PRIVATE STAN_MATH)
    # add_definitions(-std=c++1y -pthread -DGTEST_HAS_PTHREAD=0 -D_REENTRANT -Wno-ignored-attributes -Wno-sign-compare -DBOOST_DISABLE_ASSERTS)
    # add_definitions(-std=c++1y -pthread -DGTEST_HAS_PTHREAD=0 -D_REENTRANT -Wno-ignored-attributes -Wno-sign-compare -DBOOST_DISABLE_ASSERTS 
    # -I /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/tbb_2019_U8/include -O3  -I /home/eric/hybrid-sim-exp/tds/third_party/stan_math/ 
    # -I /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/eigen_3.3.7 -I /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/boost_1.72.0 
    # -I /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/sundials_5.2.0/include    -DBOOST_DISABLE_ASSERTS          
    #   -Wl,-L,"/home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/tbb" -Wl,-rpath,"/home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/tbb" 
    #   /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/sundials_5.2.0/lib/libsundials_nvecserial.a 
    #   /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/sundials_5.2.0/lib/libsundials_cvodes.a 
    #   /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/sundials_5.2.0/lib/libsundials_idas.a 
    #   /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/sundials_5.2.0/lib/libsundials_kinsol.a  
    #   /home/eric/hybrid-sim-exp/tds/third_party/stan_math/lib/tbb/libtbb.so.2)

endif(USE_STAN)

if (USE_PAGMO)
    target_link_libraries(test_pagmo_estimation PRIVATE Pagmo::pagmo)
endif(USE_PAGMO)

if (USE_RBDL)
    add_executable(test_rbdl test_rbdl.cpp)
    set_target_properties(test_rbdl PROPERTIES FOLDER autodiff)
    target_link_libraries(test_rbdl PUBLIC pybullet_visualizer_api)
    target_link_libraries(test_rbdl PRIVATE tinyxml2 ${PYTHON_LIBRARIES})
    target_link_libraries(test_rbdl PRIVATE ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} rbdl-static)
    target_include_directories(test_rbdl PUBLIC ${BULLET_INCLUDE_DIR})
    target_compile_definitions(test_rbdl PUBLIC ${BULLET_DEFINITIONS})
    target_link_directories(test_rbdl PUBLIC ${Bullet_DIR}/${BULLET_ROOT_DIR}/${BULLET_LIBRARY_DIRS})
    target_link_libraries(test_rbdl PUBLIC BulletDynamics BulletCollision LinearMath)
endif(USE_RBDL)

else()
 message("No gtest")
endif()

add_executable(adbench adbench.cpp)
target_include_directories(adbench PRIVATE ../src)
target_link_libraries(adbench
  PRIVATE
    ${CMAKE_DL_LIBS}
    benchmark
    CppADCodeGen
    ${CERES_LIBRARIES}
)

add_executable(adtest adtest.cpp)
target_include_directories(adtest PRIVATE ../src)
target_link_libraries(adtest
  PRIVATE
    ${CMAKE_DL_LIBS}
    benchmark
    gtest_main
    gmock
    CppADCodeGen
    ${CERES_LIBRARIES}
)


add_executable(test_codegen test_codegen.cpp)
target_include_directories(test_codegen PRIVATE ../src)
target_link_libraries(test_codegen
  PRIVATE
    ${CMAKE_DL_LIBS}
    benchmark
    gtest_main
    gmock
    CppADCodeGen
    ${STAN_LIBRARIES}
    ${CERES_LIBRARIES}
)
