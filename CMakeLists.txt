cmake_minimum_required(VERSION 3.16)
project(naive_infer)
set(CMAKE_CXX_STANDARD 17)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
aux_source_directory(./test DIR_TEST)
aux_source_directory(./source DIR_SOURCE)

add_executable(naive_infer main.cpp ${DIR_TEST} ${DIR_SOURCE})
target_link_libraries(naive_infer ${link_lib} ${link_math_lib})

enable_testing()