// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/matrix.h
/// \brief add your file description here.

#pragma once

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
//template <typename Matrix>
//void print_cpp_matrix(const std::string& name, const Matrix& A)
//{
//  using Scalar = typename Matrix::Scalar;
//  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<Scalar>>::value;
//
//  if constexpr (IsSparse)
//  {
//    print_cpp_matrix(name, mkl::to_eigen<Scalar>(A));
//  }
//  else
//  {
//    print_cpp_matrix(name, A);
//  }
//}

// Print the matrix A to standard output similar to numpy output.
//template <typename Matrix>
//void print_numpy_matrix(const std::string& name, const Matrix& A)
//{
//  using Scalar = typename Matrix::Scalar;
//  static const bool IsSparse = std::is_same<Matrix, mkl::sparse_matrix_csr<Scalar>>::value;
//
//  if constexpr (IsSparse)
//  {
//    eigen::print_numpy_matrix(name, mkl::to_eigen<Scalar>(A));
//  }
//  else
//  {
//    eigen::print_numpy_matrix(name, A);
//  }
//}

} // namespace nerva

