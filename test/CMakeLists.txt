enable_testing()
    
find_package(GTest CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

if (USE_STAN)
add_executable(test_gradients test_gradients.cpp)
target_link_libraries(test_gradients PRIVATE GTest::gtest GTest::gtest_main ${STAN_LIBRARIES} ${CERES_LIBRARIES})
add_test(GradientTests test_gradients)
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

add_executable(test_ceres_estimation test_ceres_estimation.cpp)
target_link_libraries(test_ceres_estimation PRIVATE GTest::gtest GTest::gtest_main ${STAN_LIBRARIES} ${CERES_LIBRARIES})
add_test(EstimationTests test_ceres_estimation)

endif(USE_STAN)

if (USE_RBDL)
add_executable(test_rbdl test_rbdl.cpp)
set_target_properties(test_rbdl PROPERTIES FOLDER autodiff)
# target_link_libraries(test_rbdl PRIVATE opengl_window tinyxml2 ${PYTHON_LIBRARIES})
target_link_libraries(test_rbdl PUBLIC pybullet_visualizer_api)
target_link_libraries(test_rbdl PRIVATE tinyxml2 ${PYTHON_LIBRARIES})
target_link_libraries(test_rbdl PRIVATE GTest::gtest GTest::gtest_main rbdl-static)
target_include_directories(test_rbdl PUBLIC ${BULLET_INCLUDE_DIR})
target_compile_definitions(test_rbdl PUBLIC ${BULLET_DEFINITIONS})
target_link_directories(test_rbdl PUBLIC ${Bullet_DIR}/${BULLET_ROOT_DIR}/${BULLET_LIBRARY_DIRS})
target_link_libraries(test_rbdl PUBLIC BulletDynamics BulletCollision LinearMath)

endif(USE_RBDL)
