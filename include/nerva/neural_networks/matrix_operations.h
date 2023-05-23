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

namespace nerva {

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
auto div(const Matrix& X)
{
  return 1 / X.array();
}

template <class Matrix>
auto diag(const Matrix& X)
{
  return X.diagonal();
}

template <class Matrix>
auto Diag(const Matrix& x)
{
  return x.asDiagonal();
}

template <class Matrix1, class Matrix2>
auto hadamard(const Matrix1& X, const Matrix2& Y)
{
  return X.cwiseProduct(Y);
}

template <typename Matrix>
auto colwise_sum(const Matrix& X)
{
  return X.colwise().sum();
}

template <typename Matrix>
auto rowwise_sum(const Matrix& X)
{
  return X.rowwise().sum();
}

template <typename Matrix>
auto colwise_replicate(const Matrix& X, long m)
{
  return X.colwise().replicate(m);
}

template <typename Matrix>
auto rowwise_replicate(const Matrix& X, long n)
{
  return X.rowwise().replicate(n);
}

} // namespace nerva

#endif // NERVA_NEURAL_NETWORKS_MATRIX_OPERATIONS_H
