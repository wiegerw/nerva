// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mkl_matrix_test.cpp
/// \brief Tests for Intel MKL libraries.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include <Eigen/Dense>
#include "nerva/neural_networks/mkl_matrix.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include <iostream>

using namespace nerva;

using vector = Eigen::VectorXd;
using matrix = Eigen::MatrixXd;
using matrix_wrapper = Eigen::Map<matrix>;

inline
long nonzero_count(const matrix& A)
{
  return (A.array() == 0.0).count();
}

template <int MatrixLayout = Eigen::ColMajor, typename Scalar>
Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>> to_eigen(mkl::dense_matrix<Scalar>& A)
{
  assert(A.layout() == MatrixLayout);
  return { A.data(), A.rows(), A.cols() };
}

template <typename Scalar, int MatrixLayout>
void test_dense_dense_multiplication(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A,
                                     const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& B)
{
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> C = A * B;
  auto A1 = mkl::make_dense_matrix_view(A);
  auto B1 = mkl::make_dense_matrix_view(B);
  auto C1 = mkl::matrix_product(A1, B1);

  scalar epsilon = std::is_same<Scalar, double>::value ? 1e-10 : 1e-5;

  std::cout << "--- test_dense_dense_multiplication ---" << std::endl;
  CHECK_LE((C - to_eigen<MatrixLayout>(C1)).squaredNorm(), epsilon);
}

void test_dense_dense_multiplication(long m, long k, long n)
{
  using matrix1 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using matrix2 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using matrix3 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using matrix4 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  matrix1 A1 = matrix1::Random(m, k);
  matrix1 B1 = matrix1::Random(k, n);
  test_dense_dense_multiplication(A1, B1);

  matrix1 A2 = matrix2::Random(m, k);
  matrix1 B2 = matrix2::Random(k, n);
  test_dense_dense_multiplication(A2, B2);

  matrix3 A3 = matrix3::Random(m, k);
  matrix3 B3 = matrix3::Random(k, n);
  test_dense_dense_multiplication(A3, B3);

  matrix4 A4 = matrix4::Random(m, k);
  matrix4 B4 = matrix4::Random(k, n);
  test_dense_dense_multiplication(A4, B4);
}

TEST_CASE("test_mkl1")
{
  matrix A {
    {2, 3},
    {7, 4},
    {1, 8},
  };

  matrix B {
    {1, 2, 3},
    {4, 5, 6}
  };

  test_dense_dense_multiplication(A, B);
}

TEST_CASE("test_mkl2")
{
  test_dense_dense_multiplication(2, 3, 4);
  test_dense_dense_multiplication(7, 1, 10);
  test_dense_dense_multiplication(12, 8, 5);
  test_dense_dense_multiplication(1, 8, 5);
  test_dense_dense_multiplication(1, 8, 1);
}

TEST_CASE("test_empty_support")
{
  using matrix_type = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

  mkl::sparse_matrix_csr<float> A(3, 4);
  matrix_type B = to_eigen(A);
  matrix_type expected = matrix_type::Zero(3, 4);
  CHECK_EQ(B, expected);
}
