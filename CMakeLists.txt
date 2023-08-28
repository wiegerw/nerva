cmake_minimum_required(VERSION 3.16)
project(nerva_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(MKL_LINK static)

add_compile_options(
  $<$<CXX_COMPILER_ID:GNU>:-Wno-writable-strings>
  $<$<CXX_COMPILER_ID:Clang>:-Wno-writable-strings>
  $<$<CXX_COMPILER_ID:Intel>:-Wno-writable-strings>
)

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(MKL CONFIG REQUIRED)
if (UNIX)
    find_package(TBB)
endif()

include_directories(${Python_INCLUDE_DIRS})
include_directories(include)

add_compile_definitions(FMT_HEADER_ONLY EIGEN_USE_MKL_ALL)
add_compile_options(-DMKL_ILP64 -m64 -march=native)

add_library(nervalib STATIC src/logger.cpp src/utilities.cpp)
target_link_libraries(nervalib PUBLIC MKL::MKL Eigen3::Eigen Python3::Python)

add_library(nervalib_shared SHARED src/logger.cpp src/utilities.cpp)
target_link_libraries(nervalib_shared PUBLIC MKL::MKL Eigen3::Eigen Python3::Python)

pybind11_add_module(nerva src/python-bindings.cpp)
target_link_libraries(nerva LINK_PUBLIC nervalib_shared MKL::MKL Eigen3::Eigen Python3::Python)

enable_testing()
add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(tools)
