cmake_minimum_required(VERSION 3.9)

project("dw-ssfp-fit")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS mpi)
find_package(Doxygen)
find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)
find_package(Pagmo REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 3.6 COMPONENTS Interpreter REQUIRED)
find_package(sycomore REQUIRED)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE}
    -c "import mpi4py; print(mpi4py.get_include())"
    OUTPUT_VARIABLE MPI4PY_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

file(GLOB_RECURSE header_files "src/*.h")
file(GLOB_RECURSE source_files "src/*.cpp")

pybind11_add_module(dw_ssfp_fit_pybind SHARED ${source_files} ${header_files})
target_include_directories(dw_ssfp_fit_pybind PUBLIC ${MPI4PY_INCLUDES})
target_link_libraries(
    dw_ssfp_fit_pybind 
    PUBLIC 
        Boost::mpi Eigen3::Eigen MPI::MPI_CXX Pagmo::pagmo sycomore::libsycomore)
set_target_properties(dw_ssfp_fit_pybind PROPERTIES OUTPUT_NAME _dw_ssfp_fit)

file(GLOB_RECURSE python_files "src/*.py")
list(SORT python_files)

add_custom_target(
    dw_ssfp_fit_pure ${CMAKE_COMMAND} -E echo "Python files" 
    SOURCES ${python_files})

execute_process(
    COMMAND ${PYTHON_EXECUTABLE}
      -c "from distutils.sysconfig import *; print(get_python_lib(True, prefix=''))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

install(DIRECTORY DESTINATION "${PYTHON_SITE_PACKAGES}")
install(TARGETS dw_ssfp_fit_pybind DESTINATION "${PYTHON_SITE_PACKAGES}/dw_ssfp_fit")

foreach(python_file ${python_files})
    file(RELATIVE_PATH module ${CMAKE_CURRENT_SOURCE_DIR}/src ${python_file})
    get_filename_component(module ${module} PATH)
    install(FILES ${python_file} DESTINATION "${PYTHON_SITE_PACKAGES}/dw_ssfp_fit")
endforeach()

if(DOXYGEN_FOUND)
    set(DOXYGEN_USE_MATHJAX YES)
    doxygen_add_docs(doxygen WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()
