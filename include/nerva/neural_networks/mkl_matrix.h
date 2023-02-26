// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/mkl_matrix.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MKL_MATRIX_H
#define NERVA_NEURAL_NETWORKS_MKL_MATRIX_H

#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/mkl_dense_matrix.h"
#include "nerva/neural_networks/mkl_sparse_matrix.h"
#include "nerva/utilities/print.h"
#include "nerva/utilities/random.h"
#include "nerva/utilities/stopwatch.h"
#include <mkl.h>
#include <mkl_spblas.h>
#include <execution>
#include <vector>

namespace nerva::mkl {

// prototype declarations
template <typename Scalar> struct sparse_matrix_csr;
void compare_sizes(const mkl::sparse_matrix_csr<scalar>& A, const mkl::sparse_matrix_csr<scalar>& B);

inline
void compare_sizes(const mkl::sparse_matrix_csr<scalar>& A, const mkl::sparse_matrix_csr<scalar>& B)
{
  if (A.rows() != B.rows() || A.cols() != B.cols())
  {
    throw std::runtime_error("matrix sizes do not match");
  }
}

template <typename Scalar>
void fill_matrix(sparse_matrix_csr<Scalar>& A, Scalar density, std::mt19937& rng, Scalar value = 0)
{
  long nonzero_count = std::lround(density * A.rows() * A.cols());
  A = sparse_matrix_csr<Scalar>(A.cols(), A.rows(), nonzero_count, rng, value);
}

template <typename Scalar, typename Function>
void initialize_matrix(sparse_matrix_csr<Scalar>& A, Function f)
{
  for (auto& value: A.values)
  {
    value = f();
  }
}

template <typename Scalar>
void load_matrix(const std::string& filename, sparse_matrix_csr<Scalar>& A)
{
  auto m = A.rows();
  auto n = A.cols();
  eigen::matrix A1(m, n);
  eigen::load_matrix(filename, A1);

  long count = 0;
  for (long i = 0; i < m; i++)
  {
    for (long k = A.row_index[i]; k < A.row_index[i + 1]; k++)
    {
      long j = A.columns[count];
      A.values[count] = A1(i, j);
      count++;
    }
  }
}

// Performs the assignment A := B, with A, B sparse. A and B must have the same support.
template <typename Scalar>
void assign_matrix(sparse_matrix_csr<Scalar>& A, const sparse_matrix_csr<Scalar>& B)
{
  std::copy(B.values.begin(), B.values.end(), A.values.begin());
}

// Does the assignment A := alpha * A + beta * op(B) * C with B sparse and A, C dense
//
// operation_B determines whether op(B) = B or op(B) = B^T
template <typename Scalar, matrix_layout layout>
void assign_matrix_product(dense_matrix_view<Scalar, layout>& A,
                           const mkl::sparse_matrix_csr<Scalar>& B,
                           const dense_matrix_view<const Scalar, layout>& C,
                           Scalar alpha = 0,
                           Scalar beta = 1,
                           sparse_operation_t operation_B = SPARSE_OPERATION_NON_TRANSPOSE
)
{
  assert(A.rows() == (operation_B == SPARSE_OPERATION_NON_TRANSPOSE ? B.rows() : B.cols()));
  assert(A.cols() == C.cols());
  assert((operation_B == SPARSE_OPERATION_NON_TRANSPOSE ? B.cols() : B.rows()) == C.rows());

  sparse_status_t status;
  if constexpr (std::is_same<Scalar, double>::value)
  {
    if constexpr (layout == matrix_layout::column_major)
    {
      status = mkl_sparse_d_mm(operation_B, beta, B.csr, B.descr, SPARSE_LAYOUT_COLUMN_MAJOR, C.data(), A.cols(), C.rows(), alpha, A.data(), A.rows());
    }
    else
    {
      status = mkl_sparse_d_mm(operation_B, beta, B.csr, B.descr, SPARSE_LAYOUT_ROW_MAJOR, C.data(), A.cols(), C.cols(), alpha, A.data(), A.cols());
    }
  }
  else
  {
    if constexpr (layout == matrix_layout::column_major)
    {
      status = mkl_sparse_s_mm(operation_B, beta, B.csr, B.descr, SPARSE_LAYOUT_COLUMN_MAJOR, C.data(), A.cols(), C.rows(), alpha, A.data(), A.rows());
    }
    else
    {
      status = mkl_sparse_s_mm(operation_B, beta, B.csr, B.descr, SPARSE_LAYOUT_ROW_MAJOR, C.data(), A.cols(), C.cols(), alpha, A.data(), A.cols());
    }
  }

  if (status != SPARSE_STATUS_SUCCESS)
  {
    throw std::runtime_error("mkl_sparse_dense_mat_mult reported status " + std::to_string(status));
  }
}

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_MATRIX_H
