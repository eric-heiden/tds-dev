enable_testing()
    
find_package(GTest CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

if (USE_STAN)
add_executable(test_gradients test_gradients.cpp)
target_link_libraries(test_gradients PRIVATE GTest::gtest GTest::gtest_main ${STAN_LIBRARIES} ${CERES_LIBRARIES})
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

add_test(GradientTests test_gradients)
endif(USE_STAN)