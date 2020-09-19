enable_testing()
    
find_package(GTest CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

add_executable(test_gradients test_gradients.cpp)
target_link_libraries(test_gradients PRIVATE GTest::gtest GTest::gtest_main ${CERES_LIBRARIES})

add_test(GradientTests test_gradients)