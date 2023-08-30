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
#include "nerva/neural_networks/mkl_sparse_matrix.h"
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
Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>> to_eigen(mkl::dense_matrix<Scalar, MatrixLayout>& A)
{
  return { A.data(), A.rows(), A.cols() };
}

template <typename Scalar, int MatrixLayout>
void test_dense_dense_multiplication(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A,
                                     const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& B,
                                     bool A_transposed,
                                     bool B_transposed)
{
  auto A0 = A_transposed ? A.transpose() : A;
  auto B0 = B_transposed ? B.transpose() : B;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> C = A0 * B0;

  auto A1 = mkl::make_dense_matrix_view(A);
  auto B1 = mkl::make_dense_matrix_view(B);
  auto C1 = mkl::ddd_product(A1, B1, A_transposed, B_transposed);
  auto C2 = mkl::ddd_product_manual_loops(A1, B1, A_transposed, B_transposed);

  scalar epsilon = std::is_same<Scalar, double>::value ? 1e-10 : 1e-5;

  std::cout << "--- test_dense_dense_multiplication ---" << std::endl;
  eigen::print_numpy_matrix("C", C);
  eigen::print_numpy_matrix("C1", mkl::to_eigen(C1));
  eigen::print_numpy_matrix("C2", mkl::to_eigen(C2));
  CHECK_LE((C - to_eigen<MatrixLayout>(C1)).squaredNorm(), epsilon);
  CHECK_LE((C - to_eigen<MatrixLayout>(C2)).squaredNorm(), epsilon);
}

void test_dense_dense_multiplication(long m, long k, long n)
{
  using matrix1 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using matrix2 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using matrix3 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using matrix4 = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  matrix1 A1 = matrix1::Random(m, k);
  matrix1 B1 = matrix1::Random(k, n);
  test_dense_dense_multiplication(A1, B1, false, false);

  matrix1 A2 = matrix2::Random(m, k);
  matrix1 B2 = matrix2::Random(k, n);
  test_dense_dense_multiplication(A2, B2, false, false);

  matrix3 A3 = matrix3::Random(m, k);
  matrix3 B3 = matrix3::Random(k, n);
  test_dense_dense_multiplication(A3, B3, false, false);

  matrix4 A4 = matrix4::Random(m, k);
  matrix4 B4 = matrix4::Random(k, n);
  test_dense_dense_multiplication(A4, B4, false, false);

  A1 = matrix1::Random(m, k);
  B1 = matrix1::Random(n, k);
  test_dense_dense_multiplication(A1, B1, false, true);

  A2 = matrix2::Random(m, k);
  B2 = matrix2::Random(n, k);
  test_dense_dense_multiplication(A2, B2, false, true);

  A3 = matrix3::Random(m, k);
  B3 = matrix3::Random(n, k);
  test_dense_dense_multiplication(A3, B3, false, true);

  A4 = matrix4::Random(m, k);
  B4 = matrix4::Random(n, k);
  test_dense_dense_multiplication(A4, B4, false, true);

  A1 = matrix1::Random(k, m);
  B1 = matrix1::Random(k, n);
  test_dense_dense_multiplication(A1, B1, true, false);

  A2 = matrix2::Random(k, m);
  B2 = matrix2::Random(k, n);
  test_dense_dense_multiplication(A2, B2, true, false);

  A3 = matrix3::Random(k, m);
  B3 = matrix3::Random(k, n);
  test_dense_dense_multiplication(A3, B3, true, false);

  A4 = matrix4::Random(k, m);
  B4 = matrix4::Random(k, n);
  test_dense_dense_multiplication(A4, B4, true, false);

  A1 = matrix1::Random(k, m);
  B1 = matrix1::Random(n, k);
  test_dense_dense_multiplication(A1, B1, true, true);

  A2 = matrix2::Random(k, m);
  B2 = matrix2::Random(n, k);
  test_dense_dense_multiplication(A2, B2, true, true);

  A3 = matrix3::Random(k, m);
  B3 = matrix3::Random(n, k);
  test_dense_dense_multiplication(A3, B3, true, true);

  A4 = matrix4::Random(k, m);
  B4 = matrix4::Random(n, k);
  test_dense_dense_multiplication(A4, B4, true, true);
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

  test_dense_dense_multiplication(A, B, false, false);
}

TEST_CASE("test_mkl2")
{
  matrix A {
    {2, 3},
    {7, 4},
    {1, 8},
  };

  matrix B {
    {1, 4},
    {2, 5},
    {3, 6}
  };

  test_dense_dense_multiplication(A, B, false, true);
}

TEST_CASE("test_mkl3")
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

TEST_CASE("test_element_access")
{
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A {
    {2, 3},
    {7, 4},
    {1, 8},
  };

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> B {
    {2, 3},
    {7, 4},
    {1, 8},
  };

  auto A1 = mkl::make_dense_matrix_view(A);
  CHECK_EQ(A1(0, 0), 2);
  CHECK_EQ(A1(1, 0), 7);
  CHECK_EQ(A1(2, 0), 1);
  CHECK_EQ(A1(0, 1), 3);
  CHECK_EQ(A1(1, 1), 4);
  CHECK_EQ(A1(2, 1), 8);

  auto B1 = mkl::make_dense_matrix_view(B);
  CHECK_EQ(B1(0, 0), 2);
  CHECK_EQ(B1(1, 0), 7);
  CHECK_EQ(B1(2, 0), 1);
  CHECK_EQ(B1(0, 1), 3);
  CHECK_EQ(B1(1, 1), 4);
  CHECK_EQ(B1(2, 1), 8);
}

TEST_CASE("test_transposed_view")
{
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A {
    {2, 3},
    {7, 4},
    {1, 8},
  };

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> B {
    {2, 3},
    {7, 4},
    {1, 8},
  };

  auto A1 = mkl::make_dense_matrix_view(A);
  auto A2 = mkl::make_transposed_dense_matrix_view(A1);
  auto A3 = mkl::to_eigen(A2);
  eigen::print_numpy_matrix("A", A);
  eigen::print_numpy_matrix("A1", mkl::to_eigen(A1));
  eigen::print_numpy_matrix("A2", mkl::to_eigen(A2));
  eigen::print_numpy_matrix("A3", A3);
  CHECK_EQ(A.transpose(), A3);

  auto B1 = mkl::make_dense_matrix_view(B);
  auto B2 = mkl::make_transposed_dense_matrix_view(B1);
  auto B3 = mkl::to_eigen(B2);
  eigen::print_numpy_matrix("B", B);
  eigen::print_numpy_matrix("B1", mkl::to_eigen(B1));
  eigen::print_numpy_matrix("B2", mkl::to_eigen(B2));
  eigen::print_numpy_matrix("B3", B3);
  CHECK_EQ(B.transpose(), B3);
}

// Test the computation C := op(A) * B with A sparse and B, C dense.
template <int MatrixLayout>
void test_sparse_dense_multiplication(const matrix& A,
                                      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& B,
                                      bool A_transposed)
{
  std::cout << "--- test_sparse_dense_multiplication ---" << std::endl;
  using result_type = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>;

  auto A0 = A_transposed ? A.transpose() : A;
  matrix C = A0 * B;

  eigen::print_numpy_matrix("A0", A0);
  eigen::print_numpy_matrix("B", B);
  eigen::print_numpy_matrix("C", C);

  mkl::sparse_matrix_csr<double> A1 = mkl::to_csr<double>(A);
  result_type C1(C.rows(), C.cols());
  sparse_operation_t operation_A = A_transposed ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE;
  double alpha = 0;
  double beta = 1;
  mkl::dsd_product(C1, A1, B, alpha, beta, operation_A);

  scalar epsilon = 1e-10;
  eigen::print_numpy_matrix("C1", C1);

  CHECK_LE((C - C1).squaredNorm(), epsilon);
}

void test_sparse_dense_multiplication(long m, long k, long n)
{
  using matrix1 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using matrix2 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  matrix A = matrix::Random(m, k);
  matrix A_transposed = matrix::Random(k, m);

  matrix1 B1 = matrix1::Random(k, n);
  test_sparse_dense_multiplication(A, B1, false);
  test_sparse_dense_multiplication(A_transposed, B1, true);

  matrix2 B2 = matrix2::Random(k, n);
  test_sparse_dense_multiplication(A, B2, false);
  test_sparse_dense_multiplication(A_transposed, B2, true);
}

TEST_CASE("test_sparse_dense")
{
  test_sparse_dense_multiplication(2, 2, 2);
  test_sparse_dense_multiplication(2, 3, 4);
  test_sparse_dense_multiplication(7, 1, 10);
  test_sparse_dense_multiplication(12, 8, 5);
  test_sparse_dense_multiplication(1, 8, 5);
  test_sparse_dense_multiplication(1, 8, 1);
}
