// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_eigen.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MKL_EIGEN_H
#define NERVA_NEURAL_NETWORKS_MKL_EIGEN_H

#include <omp.h>
#include "fmt/format.h"
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/utilities/stopwatch.h"

namespace nerva::mkl {

template <typename Scalar, int MatrixLayout>
mkl::dense_matrix_view<Scalar> make_dense_matrix_view(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  return mkl::dense_matrix_view<Scalar>(const_cast<Scalar*>(A.data()), A.rows(), A.cols(), MatrixLayout);
}

template <typename Scalar = scalar, int MatrixLayout = eigen::default_matrix_layout>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> to_eigen(const mkl::sparse_matrix_csr<Scalar>& A)
{
  int m = A.rows();
  int n = A.cols();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> result = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>::Zero(m, n);

  const auto& A_values = A.values();
  const auto& A_col_index = A.col_index();
  const auto& A_row_index = A.row_index();

  long count = 0;
  for (long i = 0; i < m; i++)
  {
    for (long k = A_row_index[i]; k < A_row_index[i + 1]; k++)
    {
      Scalar value = A_values[count];
      long j = A_col_index[count];
      count++;
      result(i, j) = value;
    }
  }

  return result;
}

// returns a boolean matrix with the non-zero entries of A
template <typename Scalar = scalar, int MatrixLayout = eigen::default_matrix_layout>
Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> support(const mkl::sparse_matrix_csr<Scalar>& A)
{
  using int_matrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>;

  int m = A.rows();
  int n = A.cols();
  int_matrix result = int_matrix::Zero(m, n);

  const auto& A_col_index = A.col_index();
  const auto& A_row_index = A.row_index();

  long count = 0;
  for (long i = 0; i < m; i++)
  {
    for (long k = A_row_index[i]; k < A_row_index[i + 1]; k++)
    {
      long j = A_col_index[count];
      count++;
      result(i, j) = 1;
    }
  }

  return result;
}

template <typename Scalar = scalar, int MatrixLayout = eigen::default_matrix_layout>
mkl::sparse_matrix_csr<Scalar> to_csr(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A)
{
  long m = A.rows();
  long n = A.cols();
  long size = eigen::nonzero_count(A);

  std::vector<MKL_INT> row_index;
  std::vector<MKL_INT> columns;
  std::vector<Scalar> values;
  row_index.reserve(A.rows() + 1);
  columns.reserve(size);
  values.reserve(size);

  row_index.push_back(0);

  long count = 0; // the number of nonzero elements
  for (long i = 0; i < m; i++)
  {
    for (long j = 0; j < n; j++)
    {
      auto value = A(i, j);
      if (value != Scalar(0))
      {
        count++;
        columns.push_back(j);
        values.push_back(value);
      }
    }
    row_index.push_back(count);
  }

  return mkl::sparse_matrix_csr<Scalar>(m, n, row_index, columns, values);
}

// Performs the assignment A := B * C, with A sparse and B, C dense.
// N.B. Only the existing entries of A are changed.
// Note that this implementation is very slow.
template <typename EigenMatrix1, typename EigenMatrix2, typename Scalar = scalar>
void assign_matrix_product_for_loop_eigen_dot(mkl::sparse_matrix_csr<Scalar>& A,
                                              const EigenMatrix1& B,
                                              const EigenMatrix2& C
)
{
  assert(A.rows() == B.rows());
  assert(A.cols() == C.cols());
  assert(B.cols() == C.rows());

  auto m = A.rows();
  auto A_values = const_cast<Scalar*>(A.values().data());
  const auto& A_col_index = A.col_index();
  const auto& A_row_index = A.row_index();

#pragma omp parallel for
  for (long i = 0; i < m; i++)
  {
    for (long k = A_row_index[i]; k < A_row_index[i + 1]; k++)
    {
      long j = A_col_index[k];
      *A_values++ = B.row(i).dot(C.col(j));
    }
  }
  A.construct_csr();
}

// Performs the assignment A := B * C, with A sparse and B, C dense.
// N.B. Only the existing entries of A are changed.
template <typename EigenMatrix1, typename EigenMatrix2, typename Scalar = scalar>
void assign_matrix_product_for_loop_mkl_dot(mkl::sparse_matrix_csr<Scalar>& A,
                                            const EigenMatrix1& B,
                                            const EigenMatrix2& C
)
{
  assert(A.rows() == B.rows());
  assert(A.cols() == C.cols());
  assert(B.cols() == C.rows());

  auto m = B.rows();
  auto p = B.cols();
  auto n = C.cols();
  auto A_values = const_cast<Scalar*>(A.values().data());
  auto B_values = const_cast<Scalar*>(B.data());
  auto C_values = const_cast<Scalar*>(C.data());
  const auto& A_col_index = A.col_index();
  const auto& A_row_index = A.row_index();

  MKL_INT incx = 1;
  MKL_INT incy = n;
  MKL_INT N = p;

  #pragma omp parallel for
  for (long i = 0; i < m; i++)
  {
    for (long k = A_row_index[i]; k < A_row_index[i + 1]; k++)
    {
      long j = A_col_index[k];
      if constexpr (std::is_same<Scalar, float>::value)
      {
        *A_values++ = cblas_sdot(N, B_values + i * p, incx, C_values + j, incy);
      }
      else
      {
        *A_values++ = cblas_ddot(N, B_values + i * p, incx, C_values + j, incy);
      }
    }
  }
  A.construct_csr();
}

// Performs the assignment A := B * C, with A sparse and B, C dense.
// N.B. Only the existing entries of A are changed.
// Use a sequential computation to copy values to A
template <typename EigenMatrix1, typename EigenMatrix2, typename Scalar = scalar, int MatrixLayout = eigen::default_matrix_layout>
void assign_matrix_product_batch(mkl::sparse_matrix_csr<Scalar>& A,
                                 const EigenMatrix1& B,
                                 const EigenMatrix2& C,
                                 long batch_size
)
{
  assert(A.rows() == B.rows());
  assert(A.cols() == C.cols());
  assert(B.cols() == C.rows());

  long m = A.rows();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> BC;
  auto values = const_cast<Scalar*>(A.values().data());
  const auto& A_col_index = A.col_index();
  const auto& A_row_index = A.row_index();

  long i_first = 0;
  while (i_first < m)
  {
    long i_last = std::min(i_first + batch_size, m);
    auto batch = Eigen::seq(i_first, i_last - 1);
    auto Bbatch = B(batch, Eigen::all);
    BC = Bbatch * C;
    for (long i = i_first; i < i_last; i++)
    {
      for (long k = A_row_index[i]; k < A_row_index[i + 1]; k++)
      {
        long j = A_col_index[k];
        *values++ = BC(i - i_first, j);
      }
    }
    i_first = i_last;
  }
  A.construct_csr();
}

// Performs the assignment A := B * C, with A sparse and B, C dense.
// N.B. Only the existing entries of A are changed.
// Use a parallel computation to copy values to A
// N.B. This doesn't seem to work!
template <typename EigenMatrix1, typename EigenMatrix2, typename Scalar = scalar, int MatrixLayout = eigen::default_matrix_layout>
void assign_matrix_product_batch_omp_parallel_assignment(mkl::sparse_matrix_csr<Scalar>& A,
                                                         const EigenMatrix1& B,
                                                         const EigenMatrix2& C,
                                                         long batch_size
)
{
  assert(A.rows() == B.rows());
  assert(A.cols() == C.cols());
  assert(B.cols() == C.rows());

  long m = A.rows();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout> BC;
  auto A_values = const_cast<Scalar*>(A.values().data());
  const auto& A_col_index = A.col_index();
  const auto& A_row_index = A.row_index();

  long i_first = 0;
  while (i_first < m)
  {
    long i_last = std::min(i_first + batch_size, m);
    auto batch = Eigen::seq(i_first, i_last - 1);
    auto Bbatch = B(batch, Eigen::all);
    utilities::stopwatch watch;
    BC = Bbatch * C;
    std::cout << fmt::format("dense : {:6.6f}\n", watch.seconds());

    watch.reset();
    // #pragma omp for
    for (long i = i_first; i < i_last; i++)
    {
      for (long k = A_row_index[i]; k < A_row_index[i + 1]; k++)
      {
        long j = A_col_index[k];
        A_values[k] = BC(i - i_first, j);
      }
    }
    std::cout << fmt::format("assign: {:6.6f}\n", watch.seconds());
    i_first = i_last;
  }
  A.construct_csr();
}

// Does the assignment A := alpha * A + beta * op(B) * C with B sparse and A, C dense
//
// Matrix C must have column major layout.
// operation_B determines whether op(B) = B or op(B) = B^T
template <typename Scalar = scalar, int MatrixLayout = eigen::default_matrix_layout>
void assign_matrix_product(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& A,
                           const mkl::sparse_matrix_csr<Scalar>& B,
                           const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& C,
                           Scalar alpha = 0,
                           Scalar beta = 1,
                           sparse_operation_t operation_B = SPARSE_OPERATION_NON_TRANSPOSE
)
{
  auto A_view = mkl::make_dense_matrix_view(A);
  auto C_view = mkl::make_dense_matrix_view(C);
  mkl::assign_matrix_product(A_view, B, C_view, alpha, beta, operation_B);
}

// returns the L2 norm of (B - A)
template <typename Scalar = scalar, int MatrixLayout = eigen::default_matrix_layout>
Scalar l2_distance(const mkl::sparse_matrix_csr<Scalar>& A, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, MatrixLayout>& B)
{
  return (B - to_eigen(A)).squaredNorm();
}

// Does the assignment A := alpha * A + beta * B, with A, B sparse.
// A and B must have the same non-zero mask
template <typename Scalar>
void assign_matrix_sum(mkl::sparse_matrix_csr<Scalar>& A,
                       const mkl::sparse_matrix_csr<Scalar>& B,
                       Scalar alpha = 0.0,
                       Scalar beta = 1.0
)
{
  assert(A.rows() == B.rows());
  assert(A.cols() == B.cols());
  assert(A.col_index() == B.col_index());
  assert(A.row_index() == B.row_index());

  eigen::vector_map<Scalar> A1(const_cast<Scalar*>(A.values().data()), A.values().size());
  eigen::vector_map<Scalar> B1(const_cast<Scalar*>(B.values().data()), B.values().size());

  A1 = alpha * A1 + beta * B1;
  A.construct_csr();
}

// Does the assignment A := alpha * A + beta * B + gamma * C, with A, B, C sparse.
// A, B and C must have the same support
template <typename Scalar>
void assign_matrix_sum(mkl::sparse_matrix_csr<Scalar>& A,
                       const mkl::sparse_matrix_csr<Scalar>& B,
                       const mkl::sparse_matrix_csr<Scalar>& C,
                       Scalar alpha = 1.0,
                       Scalar beta = 1.0,
                       Scalar gamma = 0.0
)
{
  assert(A.rows() == B.rows());
  assert(A.rows() == C.rows());
  assert(A.cols() == B.cols());
  assert(A.cols() == C.cols());
  assert(A.values().size() == B.values().size());
  assert(A.values().size() == C.values().size());

  eigen::vector_map<Scalar> A1(const_cast<Scalar*>(A.values().data()), A.values().size());
  eigen::vector_map<Scalar> B1(const_cast<Scalar*>(B.values().data()), B.values().size());
  eigen::vector_map<Scalar> C1(const_cast<Scalar*>(C.values().data()), C.values().size());

  A1 = alpha * A1 + beta * B1 + gamma * C1;
  A.construct_csr();
}

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_EIGEN_H
