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

} // namespace nerva::mkl

#endif // NERVA_NEURAL_NETWORKS_MKL_MATRIX_H
