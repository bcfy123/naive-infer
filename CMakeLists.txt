cmake_minimum_required(VERSION 3.16)
project(naiveinfer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest pthread)
set(link_math_lib  ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# 列出所有cpp源码文件
add_executable(unit_test
        axby.cpp
        test1.cpp
        main.cpp
        tensor_create_test.cpp
        tensor.hpp
        tensor.cpp
        test2.cpp
        test_size.cpp
        fill_reshape_test.cpp
)

target_link_libraries(unit_test ${link_lib} ${link_math_lib})
enable_testing()