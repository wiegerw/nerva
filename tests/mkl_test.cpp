// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mkl_test.cpp
/// \brief Tests for Intel MKL libraries.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include <Eigen/Dense>
#include "nerva/utilities/print.h"
#include "nerva/neural_networks/mkl_eigen.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/layers.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/utilities/stopwatch.h"
#include <mkl.h>
#include <mkl_spblas.h>
#include <cassert>
#include <algorithm>
#include <iostream>

using namespace nerva;

using vector = Eigen::VectorXd;
using matrix = Eigen::MatrixXd;
using matrix_wrapper = Eigen::Map<matrix>;

inline
std::size_t nonzero_count(const matrix& A)
{
  return (A.array() == 0).count();
}

// prototypes
struct matrix_mkl;

template <typename MKLMatrix>
matrix_mkl mkl_mat_mult(MKLMatrix& A, MKLMatrix& B);

inline
void print_matrix(const std::string& name, const matrix& x)
{
  std::cout << name << " = " << x.rows() << "x" << x.cols() << "\n";
  std::cout << "{\n";
  for (long i = 0; i < x.rows(); i++)
  {
    std::cout << "  {";
    for (long j = 0; j < x.cols(); j++)
    {
      if (j != 0)
      {
        std::cout << ", ";
      }
      std::cout << x(i, j);
    }
    std::cout << (i < x.rows() - 1 ? "},\n" : "}\n");
  }
  std::cout << "}\n";
}

struct matrix_mkl
{
  int m; // number of rows
  int n; // number of columns
  double* values;

  matrix_mkl(int m_, int n_)
   : m(m_), n(n_)
  {
    values = (double*) mkl_malloc(m * n * sizeof(double), 64);
  }

  [[nodiscard]] int rows() const
  {
    return m;
  }

  [[nodiscard]] int cols() const
  {
    return n;
  }

  void fill(double value)
  {
    std::fill(values, values + m * n, value);
  }

  ~matrix_mkl()
  {
    // TODO: avoid this destructor
    mkl_free(values);
  }
};

template <typename MKLMatrix>
matrix_mkl mkl_mat_mult(MKLMatrix& A, MKLMatrix& B)
{
  assert(A.cols() == B.rows());

  // m Specifies the number of rows of the matrix A and of the matrix C.
  // n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
  // k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
  MKL_INT m = A.rows();
  MKL_INT n = B.cols();
  MKL_INT k = A.cols();

  // the result
  matrix_mkl C(m, n);
  C.fill(0); // TODO: is this needed?

  auto layout = CblasColMajor;
  auto transA = CblasNoTrans;
  auto transB = CblasNoTrans;
  double alpha = 1.0;
  double beta = 0.0;

  // layout == CblasRowMajor:
  // lda >= k
  // ldb >= n
  // ldc >= n
  //
  // layout == CblasColMajor:
  // lda >= m
  // ldb >= k
  // ldc >= m
  auto lda = layout == CblasRowMajor ? k : m;
  auto ldb = layout == CblasRowMajor ? n : k;
  auto ldc = layout == CblasRowMajor ? n : m;

  double* a = A.values;
  double* b = B.values;
  double* c = C.values;
  cblas_dgemm(layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

  return C;
}

struct matrix_mkl_view
{
  int m; // number of rows
  int n; // number of columns
  double* values;

  int rows() const
  {
    return m;
  }

  int cols() const
  {
    return n;
  }

  template <typename MKLMatrix>
  matrix_mkl operator*(const MKLMatrix& other) const
  {
    return mkl_mat_mult(*this, other);
  }

  matrix_mkl_view(int m_, int n_, double* values_)
   : m(m_), n(n_), values(values_)
  {}
};

inline
matrix_mkl_view to_mkl(matrix& A)
{
  return { static_cast<int>(A.rows()), static_cast<int>(A.cols()), A.data() };
}

inline
matrix_wrapper to_eigen(matrix_mkl& A)
{
  return { A.values, A.m, A.n };
}

//--- sparse matrices ---//
template <typename SparseMatrix, typename MKLMatrix>
matrix_mkl mkl_sparse_dense_mat_mult(SparseMatrix& A, MKLMatrix& B);

// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/appendix-a-linear-solvers-basics/sparse-matrix-storage-formats/sparse-blas-csr-matrix-storage-format.html
struct csr_matrix
{
  std::vector<MKL_INT> row_index;
  std::vector<MKL_INT> columns;
  std::vector<double> values;
  int column_size;
  sparse_matrix_t csr{};
  matrix_descr descr{};

  void construct_csrA(int rows, int cols)
  {
    // Create handle with matrix stored in CSR format
    auto status = mkl_sparse_d_create_csr(&csr,
                                          SPARSE_INDEX_BASE_ZERO,
                                          rows,
                                          cols,
                                          row_index.data(),
                                          row_index.data() + 1,
                                          columns.data(),
                                          values.data());

    if (status != SPARSE_STATUS_SUCCESS)
    {
      throw std::runtime_error("Error in mkl_sparse_d_create_csr: " + std::to_string(status));
    }
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    descr.mode = SPARSE_FILL_MODE_FULL;
  }

  csr_matrix(int rows, int cols, int nonzero_count)
   : row_index(rows + 1), columns(nonzero_count), values(nonzero_count), column_size(cols)
  {
    construct_csrA(rows, cols);
  }

  explicit csr_matrix(const matrix& A)
   : column_size(static_cast<int>(A.cols()))
  {
    long size = nonzero_count(A);
    row_index.reserve(A.rows() + 1);
    columns.reserve(size);
    values.reserve(size);

    row_index.push_back(0);

    int count = 0; // the number of nonzero elements
    for (long i = 0; i < A.rows(); i++)
    {
      for (long j = 0; j < A.cols(); j++)
      {
        double value = A(i, j);
        if (value != 0.0)
        {
          count++;
          columns.push_back(static_cast<int>(j));
          values.push_back(value);
        }
      }
      row_index.push_back(count);
    }

    construct_csrA(static_cast<int>(A.rows()), static_cast<int>(A.cols()));
  }

  int rows() const
  {
    return static_cast<int>(row_index.size() - 1);
  }

  int cols() const
  {
    return column_size;
  }

  matrix_mkl operator*(const matrix_mkl_view& other) const
  {
    return mkl_sparse_dense_mat_mult(*this, other);
  }

  std::string to_string() const
  {
    std::ostringstream out;
    out << "values: " << nerva::print_list(values) << '\n';
    out << "columns: " << nerva::print_list(columns) << '\n';
    out << "row_index: " << nerva::print_list(row_index) << '\n';
    return out.str();
  }

  ~csr_matrix()
  {
    // TODO: avoid this destructor
    mkl_sparse_destroy(csr);
  }
};

template <typename SparseMatrix, typename MKLMatrix>
matrix_mkl mkl_sparse_dense_mat_mult(SparseMatrix& A, MKLMatrix& B)
{
  assert(A.cols() == B.rows());

  // the result
  matrix_mkl C(A.rows(), B.cols());
  C.fill(0); // TODO: is this needed?

  double alpha = 1.0;
  double beta = 0.0;
  sparse_status_t status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.csr, A.descr, SPARSE_LAYOUT_COLUMN_MAJOR, B.values, C.cols(), B.rows(), beta, C.values, C.rows());

  if (status != SPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error("Error in mkl_sparse_dense_mat_mult: " + std::to_string(status));
  }

  return C;
}

void test_dense_dense_multiplication(matrix& A, matrix& B)
{
  matrix C = A * B;

  matrix_mkl_view A1 = to_mkl(A);
  matrix_mkl_view B1 = to_mkl(B);
  auto C1 = A1 * B1;

  std::cout << "--- test_dense_dense_multiplication ---" << std::endl;
  print_matrix("A", A);
  print_matrix("B", B);
  print_matrix("C", C);
  print_matrix("C1", to_eigen(C1));

  CHECK(C == to_eigen(C1));
}

void test_sparse_dense_multiplication(matrix& A, matrix& B)
{
  matrix C = A * B;

  csr_matrix A1(A);
  matrix_mkl_view B1 = to_mkl(B);

  std::cout << "--- test_sparse_dense_multiplication ---" << std::endl;
  print_matrix("A", A);
  print_matrix("B", B);
  print_matrix("C", C);
  std::cout << "matrix A1" << std::endl;
  std::cout << A1.to_string() << std::endl;

  matrix_mkl C1 = A1 * B1;
  print_matrix("C1", to_eigen(C1));

  CHECK(C == to_eigen(C1));
}

TEST_CASE("test_mkl1")
{
  matrix A {
    {2, 0},
    {0, 3}
  };

  matrix B {
    {1, 2},
    {3, 4}
  };

  test_dense_dense_multiplication(A, B);
}

TEST_CASE("test_mkl2")
{
  matrix A {
    {2, 0},
    {0, 3}
  };

  matrix B {
    {1, 2, 3},
    {4, 5, 6}
  };

  test_dense_dense_multiplication(A, B);
}

TEST_CASE("test_mkl3")
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

TEST_CASE("test_csr_matrix_construction")
{
  matrix A {
    {1, -1, 0, -3, 0},
    {-2, 5, 0, 0, 0},
    {0, 0, 4, 6, 4},
    {-4, 0, 2, 7, 0},
    {0, 8, 0, 0, -5}
  };

  print_matrix("A", A);
  csr_matrix A1(A);
  std::cout << A1.to_string() << std::endl;

  std::vector<double> values = { 1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5 };
  std::vector<MKL_INT> columns = { 0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4 };
  std::vector<MKL_INT> row_index = { 0, 3, 5, 8, 11, 13 };

  CHECK_EQ(values, A1.values);
  CHECK_EQ(columns, A1.columns);
  CHECK_EQ(row_index, A1.row_index);
}

TEST_CASE("test_csr_matrix1")
{
  matrix A {
    {2, 0},
    {0, 3}
  };

  matrix B {
    {1, 2},
    {3, 4}
  };

  test_sparse_dense_multiplication(A, B);
}

TEST_CASE("test_csr_matrix2")
{
  matrix A {
    {2, 0},
    {0, 3}
  };

  matrix B {
    {1, 2, 3},
    {4, 5, 6}
  };

  test_sparse_dense_multiplication(A, B);
}

TEST_CASE("test_csr_matrix_standalone")
{
  std::cout << "--- test_csr_matrix_standalone ---" << std::endl;

  //  sparse matrix A {
  //    {0, 3, 1},
  //    {2, 0, 4}
  //  }
  int A_rows = 2;
  int A_cols = 3;
  std::array<double, 4> A_values = { 3, 1, 2, 4 };
  std::array<MKL_INT, 4> A_columns = { 1, 2, 0, 2 };
  std::array<MKL_INT, 3> A_row_index = { 0, 2, 4 };

  //  dense matrix B {
  //    {1},
  //    {4},
  //    {5}
  //  }
  int B_rows = 3;
  int B_cols [[maybe_unused]] = 1;
  std::array<double, 3> B_values = { 1, 4, 5 };

  // dense matrix C {
  //    {17},
  //    {22}
  //  }
  int C_rows = 2;
  int C_cols = 1;
  std::array<double, 2> C_values = { 0, 0 };

  // compute C = A * B
  sparse_matrix_t A_csr{};
  auto status1 = mkl_sparse_d_create_csr(&A_csr,
                                         SPARSE_INDEX_BASE_ZERO,
                                         A_rows,
                                         A_cols,
                                         A_row_index.data(),
                                         A_row_index.data() + 1,
                                         A_columns.data(),
                                         A_values.data());

  if (status1 != SPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error("Error in mkl_sparse_d_create_csr: " + std::to_string(status1));
  }

  matrix_descr A_descr{};
  A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;

  double alpha = 1.0;
  double beta = 0.0;

  auto status2 = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_csr, A_descr, SPARSE_LAYOUT_COLUMN_MAJOR, B_values.data(), C_cols, B_rows, beta, C_values.data(), C_rows);
  if (status2 != SPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error("Error in mkl_sparse_d_mm: " + std::to_string(status2));
  }

  CHECK_EQ(C_values[0], 17);
  CHECK_EQ(C_values[1], 22);
}

TEST_CASE("test_csr_matrix3")
{
  matrix A {
    {0, 3, 1},
    {2, 0, 4}
  };

  matrix B {
    {1},
    {4},
    {5}
  };

  // create sparse matrix for A
  std::vector<MKL_INT> A_row_index;
  std::vector<MKL_INT> A_columns;
  std::vector<double> A_values;
  sparse_matrix_t A_csr{};
  matrix_descr A_descr{};

  long size = nonzero_count(A);
  A_row_index.reserve(A.rows() + 1);
  A_columns.reserve(size);
  A_values.reserve(size);

  A_row_index.push_back(0);
  int count = 0; // the number of nonzero elements
  for (long i = 0; i < A.rows(); i++)
  {
    for (long j = 0; j < A.cols(); j++)
    {
      double value = A(i, j);
      if (value != 0.0)
      {
        count++;
        A_columns.push_back(static_cast<int>(j));
        A_values.push_back(value);
      }
    }
    A_row_index.push_back(count);
  }

  std::cout << "A_values: " << nerva::print_list(A_values) << '\n';
  std::cout << "A_columns: " << nerva::print_list(A_columns) << '\n';
  std::cout << "A_row_index: " << nerva::print_list(A_row_index) << '\n';

  // Create handle with matrix stored in CSR format
  auto status1 = mkl_sparse_d_create_csr(&A_csr,
                                         SPARSE_INDEX_BASE_ZERO,
                                         static_cast<int>(A.rows()),
                                         static_cast<int>(A.cols()),
                                         A_row_index.data(),
                                         A_row_index.data() + 1,
                                         A_columns.data(),
                                         A_values.data());

  if (status1 != SPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error("Error in mkl_sparse_d_create_csr: " + std::to_string(status1));
  }
  A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  A_descr.diag = SPARSE_DIAG_NON_UNIT;
  A_descr.mode = SPARSE_FILL_MODE_FULL;

  matrix_mkl_view B1 = to_mkl(B);

  int C_rows = 2;
  int C_cols = 1;
  std::array<double, 2> C_values = { 0, 0 };

  double alpha = 1.0;
  double beta = 0.0;

  auto status2 = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_csr, A_descr, SPARSE_LAYOUT_COLUMN_MAJOR, B1.values, C_cols, B1.rows(), beta, C_values.data(), C_rows);
  if (status2 != SPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error("Error in mkl_sparse_d_mm: " + std::to_string(status2));
  }

  CHECK_EQ(C_values[0], 17);
  CHECK_EQ(C_values[1], 22);
}

TEST_CASE("test_csr_matrix3a")
{
  matrix A {
    {0, 3, 1},
    {2, 0, 4}
  };

  matrix B {
    {1},
    {4},
    {5}
  };

  test_sparse_dense_multiplication(A, B);
}

TEST_CASE("test_csr_matrix4")
{
  matrix A {
    {0, 3},
    {0, 4},
    {1, 5}
  };

  matrix B {
    {1},
    {4}
  };

  test_sparse_dense_multiplication(A, B);
}

TEST_CASE("test_csr_matrix4")
{
  matrix A {
    {2, 3},
    {0, 4},
    {1, 0},
  };

  matrix B {
    {1, 2, 3},
    {4, 5, 6}
  };

  test_sparse_dense_multiplication(A, B);
}

inline
void check_csr()
{
  // sparse matrix A
  sparse_matrix_t A_csr;
  MKL_INT A_rows = 4;
  MKL_INT A_cols = 3;
  MKL_INT A_row_index[5] = {0, 1, 1, 3, 4};
  MKL_INT A_columns[4] = {1, 0, 2, 0};
  float A_values[4] = {1, 2, 3, 4};
  mkl_sparse_s_create_csr(&A_csr,
                          SPARSE_INDEX_BASE_ZERO,
                          A_rows,
                          A_cols,
                          A_row_index,
                          A_row_index + 1,
                          A_columns,
                          A_values);
  matrix_descr A_descr{SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT};

  // dense matrix B
  MKL_INT B_rows = 3;
  // MKL_INT B_cols = 1;
  float B_values[4] = {1, 2, 3};

  // dense matrix C
  MKL_INT C_rows = 4;
  MKL_INT C_cols = 1;
  float* C_values = new float[4];

  float alpha = 0;
  float beta = 1;

  // compute C = alpha * A * B + beta * C
  sparse_status_t status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, beta, A_csr, A_descr, SPARSE_LAYOUT_COLUMN_MAJOR, B_values, C_cols, B_rows, alpha, C_values, C_rows);
  if (status != SPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error("mkl_sparse_dense_mat_mult reported status " + std::to_string(status));
  }

  float expected[4] = {2, 0, 11, 4};

  std::cout << "expected: " << expected[0] << ' ' << expected[1] << ' ' << expected[2] << ' ' << expected[3] << std::endl;
  std::cout << "result:   " << C_values[0] << ' ' << C_values[1] << ' ' << C_values[2] << ' ' << C_values[3] << std::endl;

  float error = 0;
  for (int i = 0; i < 4; i++)
  {
    error += (C_values[i] - expected[i]) * (C_values[i] - expected[i]);
  }
  if (error > 0.00001)
  {
    throw std::runtime_error("unexpected result");
  }
}

TEST_CASE("test_csr2")
{
  std::cout << "--- test_csr2 ---" << std::endl;
  check_csr();
}

// Compute C = A * B, where
// A is a sparse m x k matrix
// B is a dense  k x n matrix
// C is a dense  m x n matrix
// 0 <= density <= 1 is the density of A
void test_matrix_multiplication(long m, long k, long n, scalar density)
{
  std::cout << "--- test multiplication m = " << m << " k = " << k << " n = " << n << " density = " << density << std::endl;
  scalar a = -10;
  scalar b = 10;

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  eigen::matrix A(m, k);
  eigen::fill_matrix_random(A, density, a, b, rng);

  eigen::matrix B(k, n);
  eigen::fill_matrix_random(B, 1, a, b, rng);

  eigen::matrix C(m, n);
  eigen::matrix C1(m, n);
  eigen::matrix C2(m, n);

  mkl::sparse_matrix_csr<scalar> A1 = mkl::to_csr<scalar>(A);
  CHECK_EQ(mkl::to_eigen(A1), A);

  eigen::sparse_matrix A2 = eigen::to_sparse(A);
  CHECK_EQ(eigen::matrix(A2), A);

  utilities::stopwatch watch;
  C = A * B;
  std::cout << "C = A * B " << watch.seconds() << "s" << std::endl;

  watch.reset();
  mkl::dsd_product(C1, A1, B);  // C1 := A1 * B1
  std::cout << "C1 = A1 * B " << watch.seconds() << "s" << std::endl;

  watch.reset();
  C2 = A2 * B;
  std::cout << "C2 = A2 * B " << watch.seconds() << "s" << std::endl;

  scalar epsilon = std::is_same<scalar, double>::value ? 1e-10 : 1.0;
  CHECK_LE((C - C1).squaredNorm(), epsilon);
  CHECK_LE((C - C2).squaredNorm(), epsilon);
}

// Compute C = alpha * A + B, where
// A is a sparse m x n matrix
// B is a dense  m x n matrix
// 0 <= density <= 1 is the density of A
void test_matrix_addition(long m, long k, long n, scalar density)
{
  std::cout << "--- test addition m = " << m << " k = " << k << " n = " << n << " density = " << density << std::endl;
  scalar a = -10;
  scalar b = 10;

  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  eigen::matrix A(m, n);
  eigen::fill_matrix_random(A, density, a, b, rng);

  eigen::matrix B = A;

  eigen::matrix C(m, n);

  mkl::sparse_matrix_csr<scalar> A1 = mkl::to_csr<scalar>(A);
  CHECK_EQ(mkl::to_eigen(A1), A);

  mkl::sparse_matrix_csr<scalar> B1 = mkl::to_csr<scalar>(B);
  CHECK_EQ(mkl::to_eigen(B1), B);

  mkl::sparse_matrix_csr<scalar> C1 = A1;

  eigen::sparse_matrix A2 = eigen::to_sparse(A);
  CHECK_EQ(eigen::matrix(A2), A);

  eigen::sparse_matrix B2 = eigen::to_sparse(B);
  CHECK_EQ(eigen::matrix(B2), B);

  eigen::sparse_matrix C2 = A2;

  utilities::stopwatch watch;
  C = A + B;
  std::cout << "C = A + B " << watch.seconds() << "s" << std::endl;

  watch.reset();
  mkl::sss_sum(C1, A1, B1); // C1 := A1 * B
  std::cout << "C1 = A1 + B1 " << watch.seconds() << "s" << std::endl;

  watch.reset();
  C2 = A2 + B2;
  std::cout << "C2 = A2 + B2 " << watch.seconds() << "s" << std::endl;

  scalar epsilon = std::is_same<scalar, double>::value ? 1e-10 : 1.0;
  CHECK_LE((C - to_eigen(C1)).squaredNorm(), epsilon);
  CHECK_LE((C - C2).squaredNorm(), epsilon);
}

TEST_CASE("test_multiplication")
{
  test_matrix_multiplication(1000, 900, 1200, 0.1);
  test_matrix_multiplication(1000, 900, 1200, 0.5);
  test_matrix_multiplication(1000, 900, 1200, 1.0);
}

TEST_CASE("test_addition")
{
  test_matrix_addition(1000, 900, 1200, 0.1);
  test_matrix_addition(1000, 900, 1200, 0.5);
  test_matrix_addition(1000, 900, 1200, 1.0);
}

template <typename Scalar, typename Function = zero<Scalar>>
mkl::sparse_matrix_csr<Scalar> make_csr_matrix(std::size_t rows, std::size_t columns, double density, std::mt19937& rng, Scalar value)
{
std::cout << "=== make_csr_matrix " << rows << 'x' << columns << std::endl;
  std::size_t nonzero_count = std::lround(density * rows * columns);
  return mkl::make_random_matrix<Scalar>(rows, columns, nonzero_count, rng, [value]() { return value; });
}

TEST_CASE("test_fill_matrix")
{
  auto seed = std::random_device{}();
  std::mt19937 rng{seed};

  long D = 4;   // input size
  long K = 6;   // output size
  long Q = 2;   // batch size
  double density = 1;

  mkl::sparse_matrix_csr<scalar> W = make_csr_matrix(6, 4, density, rng, scalar(2));
  mkl::sparse_matrix_csr<scalar> W1 = W;

  mkl::sparse_matrix_csr<scalar> W2;
  W2 = W;

  mkl::sparse_matrix_csr<scalar> X(6, 4);
  X = make_csr_matrix(6, 4, density, rng, scalar(2));

  sparse_linear_layer layer(D, K, Q);
  set_support_random(layer, density, rng);
}

struct counter
{
  int i = 1;

  int operator()()
  {
    return i++;
  }
};

TEST_CASE("test_assign_values")
{
  eigen::matrix A {
    {0, 3, 1},
    {2, 0, 4}
  };

  mkl::sparse_matrix_csr<scalar> A1 = mkl::to_csr<scalar>(A);
  std::vector<scalar> A_values = {3, 1, 2, 4};
  std::cout << "A1.values = " << print_list(A1.values()) << std::endl;
  CHECK_EQ(A_values, A1.values());
  eigen::matrix A2 = mkl::to_eigen(A1);
  print_cpp_matrix("A2", A2);
  CHECK_EQ(A, A2);

  eigen::matrix B {
    {0, 1, 2},
    {3, 0, 4}
  };
  mkl::initialize_matrix(A1, counter());
  A2 = mkl::to_eigen(A1);
  print_cpp_matrix("A2", A2);
  CHECK_EQ(B, A2);
}

TEST_CASE("test_eigen_layout")
{
  eigen::matrix A{
    {1, 2, 3},
    {4, 5, 6}};

  CHECK_EQ(2, A.rows());
  CHECK_EQ(3, A.cols());
  CHECK_EQ(3, A(0, 2));

  auto data = A.data();
  std::vector<scalar> v(data, data + 6);
  std::cout << "v = " << print_list(v) << std::endl;
  if constexpr (eigen::default_matrix_layout == Eigen::ColMajor)
  {
    std::vector<scalar> v_expected = {1, 4, 2, 5, 3, 6};
    CHECK_EQ(v_expected, v);
  }
  else
  {
    std::vector<scalar> v_expected = {1, 2, 3, 4, 5, 6};
    CHECK_EQ(v_expected, v);
  }
}

TEST_CASE("test_assign_matrix_product")
{
  eigen::matrix B {
    {1, 2, 3},
    {4, 5, 6},
    {5, 5, 2},
    {9, 1, 3},
    {1, 5, 1}
  };

  eigen::matrix C {
    {7, 6, 3},
    {1, 1, 8},
    {2, 4, 6}};

  eigen::matrix A = B * C;

  long m = B.rows();
  long n = C.cols();
  double density = 1.0;
  auto seed = std::random_device{}();
  std::mt19937 rng{seed};
  mkl::sparse_matrix_csr<scalar> A1 = make_csr_matrix(m, n, density, rng, scalar(0));
  mkl::sdd_product_batch(A1, B, C, 2);

  print_cpp_matrix("A", A);
  print_cpp_matrix("A1", mkl::to_eigen(A1));
  CHECK_EQ(A, mkl::to_eigen(A1));
}
