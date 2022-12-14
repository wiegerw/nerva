cmake_minimum_required(VERSION 3.16)
project(nerva LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(MKL_LINK static)

find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(OpenMP)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(MKL CONFIG REQUIRED)

if (UNIX)
    find_package(TBB REQUIRED)
endif()

include_directories(include)

add_compile_definitions(FMT_HEADER_ONLY)
if (UNIX)
    add_compile_options(-DMKL_ILP64 -m64 -march=native)
endif()

add_library(nervalib src/logger.cpp src/utilities.cpp)

pybind11_add_module(nerva src/python-bindings.cpp)
target_link_libraries(nerva LINK_PUBLIC nervalib MKL::MKL Eigen3::Eigen)

enable_testing()
add_subdirectory(tests)
