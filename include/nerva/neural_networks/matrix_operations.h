// Copyright: Wieger Wesselink 2023
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/matrix_operations.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_MATRIX_OPERATIONS_H
#define NERVA_NEURAL_NETWORKS_MATRIX_OPERATIONS_H

namespace nerva::eigen {

template <class Matrix>
bool is_column_vector(const Matrix& x)
{
  return x.cols() == 1;
}

template <class Matrix>
bool is_row_vector(const Matrix& x)
{
  return x.rows() == 1;
}

template <class Matrix>
bool is_square(const Matrix& X)
{
  return X.rows() == X.cols();
}

template <class Matrix>
auto exp(const Matrix& X)
{
  return X.array().exp();
}

template <class Matrix>
auto log(const Matrix& X)
{
  return X.array().log();
}

template <class Matrix>
auto inverse(const Matrix& X)
{
  return X.array().inverse();
}

template <class Matrix>
auto diag(const Matrix& X)
{
  assert(is_square(X));
  return X.diagonal();
}

template <class Matrix>
auto Diag(const Matrix& x)
{
  assert(is_column_vector(x) || is_row_vector(x));
  return x.asDiagonal();
}

template <class Matrix1, class Matrix2>
auto hadamard(const Matrix1& X, const Matrix2& Y)
{
  return X.cwiseProduct(Y);
}

template <typename Matrix>
auto sum_columns(const Matrix& X)
{
  return X.colwise().sum();
}

template <typename Matrix>
auto sum_rows(const Matrix& X)
{
  return X.rowwise().sum();
}

template <typename Matrix>
auto repeat_row(const Matrix& x, long m)
{
  assert(is_row_vector(x));
  return x.colwise().replicate(m);
}

template <typename Matrix>
auto repeat_column(const Matrix& x, long n)
{
  assert(is_column_vector(x));
  return x.rowwise().replicate(n);
}

} // namespace nerva::eigen

#endif // NERVA_NEURAL_NETWORKS_MATRIX_OPERATIONS_H
