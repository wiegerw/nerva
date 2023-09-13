// Copyright: Wieger Wesselink 2021 - present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file eigen_test.cpp
/// \brief Tests for the Eigen library.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <vector>

using matrix = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
using vector = Eigen::VectorXd;
using sparse_matrix_type = Eigen::SparseMatrix<double>;

// Define a custom macro similar to BOOST_CHECK_SMALL
#define DOCTEST_CHECK_SMALL(val, tol)                                      \
    DOCTEST_CHECK_MESSAGE(std::fabs(val) <= tol,                          \
                          "Expected: " #val " to be within " #tol " of 0, but it's " << val)


TEST_CASE("test_replicate")
{
  vector x {{1, 2, 3}};
  auto y = x.rowwise().replicate(4);

  matrix expected {
    {1, 1, 1, 1},
    {2, 2, 2, 2},
    {3, 3, 3, 3}
  };

  DOCTEST_CHECK_EQ(expected, y);
}

TEST_CASE("test_rowwise_sum")
{
  matrix A(2, 4);
  A << 1, 2, 6, 9,
       3, 2, 7, 2;

  auto y = A.rowwise().sum();
  vector y_expected(2);
  y_expected << 18, 14;
  DOCTEST_CHECK_EQ(y_expected, y);

  auto z = A.rowwise().sum() / 2;
  vector z_expected(2);
  z_expected << 9, 7;
  DOCTEST_CHECK_EQ(z_expected, z);
}

TEST_CASE("test_slice1")
{
  matrix A(2, 4);
  A << 1, 2, 6, 9,
       3, 2, 7, 2;

  matrix B(2, 2);
  B << 2, 6,
       2, 7;

  matrix C = A(Eigen::all, Eigen::seqN(1, 2));

  DOCTEST_CHECK_EQ(B, C);
}

class slice
{
  private:
  std::vector<long>::const_iterator m_first;
  long m_size;

  public:
  slice(std::vector<long>::const_iterator first, long size)
      : m_first(first), m_size(size)
  {}

  [[nodiscard]] Eigen::Index size() const
  {
    return m_size;
  }

  Eigen::Index operator[] (Eigen::Index i) const
  {
    return *(m_first + i);
  }
};

TEST_CASE("test_slice2")
{
  matrix A(2, 4);
  A << 1, 2, 6, 9,
       3, 2, 7, 2;

  matrix B(2, 3);
  B << 2, 9, 1,
       2, 2, 3;

  std::vector<long> I = {1, 3, 0, 2};
  matrix C = A(Eigen::all, slice(I.begin(), 3));

  DOCTEST_CHECK_EQ(B, C);
}

TEST_CASE("test_norm")
{
  matrix A(2, 4);
  A << 1, 2, 6, 9,
       3, 2, 7, 2;

  auto t1 = A.colwise().squaredNorm().sum();
  auto t2 = A.squaredNorm();
  auto expected = 1 + 9 + 4 + 4 + 36 + 49 + 81 + 4;

  DOCTEST_CHECK_EQ(expected, t1);
  DOCTEST_CHECK_EQ(expected, t2);
}

void f1(const vector& x)
{
  std::cout << "f1 " << x << std::endl;
}

void f1(const matrix& x)
{
  std::cout << "f1 " << x << std::endl;
}

TEST_CASE("test_overload")
{
  vector x(3);
  f1(x);
  matrix A(3, 3);
  f1(A);
}

TEST_CASE("test_mean")
{
  matrix x {
    {1, 2, 3},
    {3, 4, 1},
  };

  vector expected {
    {2, 3, 2},
  };

  matrix y = x.colwise().mean();
  std::cout << "mean(x) = " << y << std::endl;
  DOCTEST_CHECK_SMALL((expected.transpose() - y).squaredNorm(), 1e-10);
}

inline
matrix standardize_column_data1(const matrix& x, double epsilon = 1e-20)
{
  matrix x_minus_mean = x.colwise() - x.rowwise().mean();
  auto stddev = x_minus_mean.array().square().rowwise().sum() / x.cols();
  std::cout << "xminus\n" << x_minus_mean << std::endl;
  std::cout << "stddev\n" << (stddev + epsilon).sqrt() << std::endl;
  return x_minus_mean.array().colwise() / (stddev + epsilon).sqrt();
}

inline
matrix standardize_column_data2(const matrix& x, double epsilon = 1e-20)
{
  matrix x_minus_mean = x.colwise() - x.rowwise().mean();
  auto stddev = (x_minus_mean * x_minus_mean.transpose()).diagonal().array() / x.cols();
  std::cout << "xminus\n" << x_minus_mean << std::endl;
  std::cout << "stddev\n" << (stddev + epsilon).sqrt() << std::endl;
  return x_minus_mean.array().colwise() / (stddev + epsilon).sqrt();
}

TEST_CASE("test_standardize")
{
  std::cout << "test_standardize" << std::endl;

  matrix x {
    {1, 2, 3},
    {3, 7, 2},
  };

  matrix expected {
    {-1.22474487, 0.        ,  1.22474487},
    {-0.46291005, 1.38873015, -0.9258201},
  };

  matrix y1 = standardize_column_data1(x);
  std::cout << "y1=\n" << y1 << std::endl;
  DOCTEST_CHECK_SMALL((expected - y1).squaredNorm(), 1e-10);

  matrix y2 = standardize_column_data2(x);
  std::cout << "y2=\n" << y2 << std::endl;
  DOCTEST_CHECK_SMALL((expected - y2).squaredNorm(), 1e-10);
}

TEST_CASE("test_divide1")
{
  std::cout << "test_divide" << std::endl;

  matrix x {
    {2, 4, 6},
    {3, 12, 15},
  };

  vector b {
    {2},
    {3}
  };

  matrix expected {
    {1, 2, 3},
    {1, 4, 5},
  };

  matrix y = x.array().colwise() / b.array();
  std::cout << "y=\n" << y << std::endl;

  DOCTEST_CHECK_SMALL((expected - y).squaredNorm(), 1e-10);
}

TEST_CASE("test_sigma_divide")
{
  std::cout << "test_sigma_divide" << std::endl;

  matrix X {
    {6, 8, 10},
    {3, 12, 15},
  };

  vector Sigma {
    {4},
    {9}
  };

  matrix expected {
    {3, 4, 5},
    {1, 4, 5},
  };

  matrix Y = (Sigma.unaryExpr([](double t) { return 1.0 / std::sqrt(t); })).asDiagonal() * X;

  std::cout << "Diag=\n" << Sigma.array().sqrt() << std::endl;
  std::cout << "Y=\n" << Y << std::endl;

  DOCTEST_CHECK_SMALL((expected - Y).squaredNorm(), 1e-10);
}

TEST_CASE("test_diag_mult")
{
  std::cout << "test_diag_mult" << std::endl;

  matrix X {
    {6, 8, 10},
    {3, 12, 15},
  };

  vector y {
    {2},
    {3}
  };

  matrix expected {
    {12, 16, 20},
    {9, 36, 45},
  };

  matrix Y = X.array().colwise() * y.array();
  std::cout << "Y=\n" << Y << std::endl;

  DOCTEST_CHECK_SMALL((expected - Y).squaredNorm(), 1e-10);
}

TEST_CASE("test_multiply1")
{
  std::cout << "test_multiply1" << std::endl;

  matrix X {
    {1, 2, 3},
    {3, 4, 5},
  };

  vector b {
    {2},
    {3}
  };

  matrix expected {
    {2, 4, 6},
    {9, 12, 15},
  };

  matrix Y = X.array().colwise() * b.array();
  std::cout << "Y=\n" << Y << std::endl;
  DOCTEST_CHECK_SMALL((expected - Y).squaredNorm(), 1e-10);

  auto n = X.cols();
  Y = b.rowwise().replicate(n).cwiseProduct(X);
  std::cout << "Y=\n" << Y << std::endl;
  DOCTEST_CHECK_SMALL((expected - Y).squaredNorm(), 1e-10);
}

TEST_CASE("test_sparse")
{
  std::cout << "test_sparse" << std::endl;

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList = { T(0, 0, 5), T(0, 1, 3), T(1, 1, 2) };
  sparse_matrix_type A(2, 2);
  A.setFromTriplets(tripletList.begin(), tripletList.end());

  matrix A1 = A;

  matrix A2 {
    {5, 3},
    {0, 2},
  };

  matrix x {
    {2},
    {3},
  };

  matrix y {
    {3, 7}
  };

  sparse_matrix_type B2 = A.cwiseProduct(A2);
  sparse_matrix_type B3 = A.unaryExpr([](double t) { return t + 1; });
  sparse_matrix_type B4 = A.unaryExpr([](double t) { return double(1); }).cwiseProduct(x * y);

  matrix C1 {
    {6, 14},
    {9, 21},
  };

  matrix C2 {
    {25, 9},
    {0, 4},
  };

  matrix C3 {
    {6, 4},
    {0, 3},
  };

  matrix C4 = x * y;
  C4(1, 0) = 0;

  std::cout << "B2 =\n" << B2 << std::endl;
  std::cout << "B3 =\n" << B3 << std::endl;
  std::cout << "B4 =\n" << B4 << std::endl;
  std::cout << "C4 =\n" << C4 << std::endl;

  CHECK(A1 == A2);
  CHECK(matrix(B2) == C2);
  CHECK(matrix(B3) == C3);
  CHECK(matrix(B4) == C4);

  // check if operations are truly sparse
  matrix V(100000, 1);
  V.array() = 3;

  matrix W(1, 100000);
  W.array() = 4;

  sparse_matrix_type Z(100000, 100000);
  Z.setFromTriplets(tripletList.begin(), tripletList.end());
  std::cout << "assigning matrix" << std::endl;
  try
  {
    // Z = Z.unaryExpr([](double t) { return double(1); }).cwiseProduct(V * W);
    Z = Z.cwiseProduct(V * W); // unfortunately the product V * W is explicitly computed
    std::cout << "done" << std::endl;
  }
  catch (std::bad_alloc&)
  {
    std::cout << "allocation failed!" << std::endl;
  }
}

TEST_CASE("test_transpose")
{
  std::cout << "test_transpose" << std::endl;

  using matrix_rowmajor = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
  using matrix_colmajor = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>;

  matrix_rowmajor X {
    {1, 2, 3},
    {4, 5, 6},
  };

  matrix_rowmajor X_transposed {
    {1, 4},
    {2, 5},
    {3, 6},
  };

  // This is a cheap way to transpose matrix X
  Eigen::Map<matrix_colmajor> Y(X.data(), X.cols(), X.rows());

  DOCTEST_CHECK_EQ(X.transpose(), X_transposed);
  DOCTEST_CHECK_EQ(Y, X_transposed);
}

