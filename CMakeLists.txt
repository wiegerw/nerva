cmake_minimum_required(VERSION 3.16)
project(nerva LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(MKL_LINK static)

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(MKL CONFIG REQUIRED)
if (UNIX)
    find_package(TBB)
endif()

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include)

add_compile_definitions(FMT_HEADER_ONLY EIGEN_USE_MKL_ALL)

# These flags are advised by the Intel oneAPI Math Kernel Library Link Line Advisor,
# but they are not set automatically. Note that -DMKL_ILP64 doesn't work with clang.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-DMKL_LP64 -m64 -march=native)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-DMKL_ILP64 -m64 -march=native)
endif()

add_library(nervalib src/logger.cpp src/utilities.cpp)

pybind11_add_module(nerva src/python-bindings.cpp)
target_link_libraries(nerva LINK_PUBLIC nervalib MKL::MKL Eigen3::Eigen)

enable_testing()
add_subdirectory(tests)
add_subdirectory(tools)
