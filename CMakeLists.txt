cmake_minimum_required(VERSION 3.5)

project("dw-ssfp-fit")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(Pagmo REQUIRED)
find_package(pybind11 REQUIRED)

file(GLOB_RECURSE header_files "src/*.h")
file(GLOB_RECURSE source_files "src/*.cpp")

pybind11_add_module(dw_ssfp_fit SHARED ${source_files} ${header_files})
target_link_libraries(
    dw_ssfp_fit 
    PUBLIC
        Eigen3::Eigen Pagmo::pagmo)
