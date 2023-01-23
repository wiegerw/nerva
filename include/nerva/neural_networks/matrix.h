// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/matrix.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MATRIX_H
#define NERVA_NEURAL_NETWORKS_MATRIX_H

#include "nerva/neural_networks/mkl_eigen.h"

namespace nerva {

// Use the generator f to assign values to the coefficients of the matrix A.
template <typename Matrix, typename Function>
void initialize_matrix(Matrix& A, Function f)
{
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  if constexpr (IsSparse)
  {
    mkl::initialize_matrix(A, f);
  }
  else
  {
    eigen::initialize_matrix(A, f);
  }
}

// Print the matrix A to standard output.
template <typename Matrix>
void print_cpp_matrix(const std::string& name, const Matrix& A)
{
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  if constexpr (IsSparse)
  {
    eigen::print_cpp_matrix(name, mkl::to_eigen<scalar>(A));
  }
  else
  {
    eigen::print_cpp_matrix(name, A);
  }
}

// Print the matrix A to standard output similar to numpy output.
template <typename Matrix>
void print_numpy_matrix(const std::string& name, const Matrix& A)
{
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  if constexpr (IsSparse)
  {
    eigen::print_numpy_matrix(name, mkl::to_eigen<scalar>(A));
  }
  else
  {
    eigen::print_numpy_matrix(name, A);
  }
}

template <typename Matrix>
void load_matrix(const std::string& filename, Matrix& A)
{
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  if constexpr (IsSparse)
  {
    mkl::load_matrix(filename, A);
  }
  else
  {
    eigen::load_matrix(filename, A);
  }
}

template <typename Matrix>
void save_matrix(const std::string& filename, const Matrix& A)
{
  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<scalar>>::value;

  if constexpr (IsSparse)
  {
    mkl::save_matrix(filename, A);
  }
  else
  {
    eigen::save_matrix(filename, A);
  }
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_MATRIX_H
