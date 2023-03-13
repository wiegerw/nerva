project(NERVA_EXAMPLES)

add_executable(inspect_npz_file "inspect_npz_file.cpp")
target_link_libraries(inspect_npz_file PRIVATE Eigen3::Eigen MKL::MKL Python3::Python)
target_include_directories(inspect_npz_file PUBLIC ".")

add_executable(mkl_slow_product "mkl_slow_product.cpp")
target_link_libraries(mkl_slow_product PRIVATE Eigen3::Eigen MKL::MKL Python3::Python)
