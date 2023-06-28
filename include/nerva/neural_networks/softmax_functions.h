// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/softmax_functions.h
/// \brief add your file description here.

#ifndef NERVA_NEURAL_NETWORKS_SOFTMAX_FUNCTIONS_H
#define NERVA_NEURAL_NETWORKS_SOFTMAX_FUNCTIONS_H

#include "nerva/neural_networks/eigen.h"
#include <cassert>

namespace nerva::eigen {

template <typename Matrix>
auto softmax_colwise(const Matrix& X)
{
  auto D = X.rows(); 
  auto E = exp(X);
  return hadamard(E, row_repeat(inverse(columns_sum(E)), D));
}

template <typename Vector>
auto softmax_colwise_jacobian(const Vector& x)
{
  assert(is_column_vector(x));
  auto y = softmax_colwise(x);
  // return Diag(y) - y * y.transpose();  TODO: unfortunately asDiagonal is broken: it does not support addition / subtractio
  return Diag(y).toDenseMatrix() - y * y.transpose();
}

template <typename Matrix>
auto stable_softmax_colwise(const Matrix& X)
{
  auto D = X.rows(); 
  auto Y = X - row_repeat(columns_max(X), D);
  auto E = exp(Y);
  return hadamard(E, row_repeat(inverse(columns_sum(E)), D));
}

template <typename Vector>
auto stable_softmax_colwise_jacobian(const Vector& x)
{
  assert(is_column_vector(x));
  auto y = stable_softmax_colwise(x);
  return Diag(y) - y * y.transpose();
}

template <typename Matrix>
auto log_softmax_colwise(const Matrix& X)
{
  auto D = X.rows();
  return X - row_repeat(log(columns_sum(exp(X))), D);
}

template <typename Matrix>
auto stable_log_softmax_colwise(const Matrix& X)
{
  auto D = X.rows();
  auto Y = X - row_repeat(columns_max(X), D);
  return Y - row_repeat(log(columns_sum(exp(Y))), D);
}

template <typename Vector>
auto log_softmax_colwise_jacobian(const Vector& x)
{
  assert(is_column_vector(x));
  auto D = x.rows();
  return identity<Vector>(D) - row_repeat(softmax_colwise(x).transpose(), D);
}

template <typename Vector>
auto stable_log_softmax_colwise_jacobian(const Vector& x)
{
  return log_softmax_colwise_jacobian(x);
}

template <typename Matrix>
auto softmax_rowwise(const Matrix& X)
{
  auto D = X.cols();
  auto E = exp(X);
  return hadamard(E, column_repeat(inverse(rows_sum(E)), D));
}

template <typename Vector>
auto softmax_rowwise_jacobian(const Vector& x)
{
  assert(is_row_vector(x));
  auto y = softmax_rowwise(x);
  // return Diag(y) - y.transpose() * y;  TODO: unfortunately asDiagonal is broken: it does not support addition / subtractio
  return Diag(y).toDenseMatrix() - y.transpose() * y;
}

template <typename Matrix>
auto stable_softmax_rowwise(const Matrix& X)
{
  auto D = X.cols();
  auto Y = X - column_repeat(rows_max(X), D);
  auto E = exp(Y);
  return hadamard(E, column_repeat(inverse(rows_sum(E)), D));
}

template <typename Vector>
auto stable_softmax_rowwise_jacobian(const Vector& x)
{
  assert(is_row_vector(x));
  auto y = stable_softmax_rowwise(x);
  return Diag(y) - y.transpose() * y;
}

template <typename Matrix>
auto log_softmax_rowwise(const Matrix& X)
{
  auto D = X.cols();
  return X - column_repeat(log(rows_sum(exp(X))), D);
}

template <typename Vector>
auto log_softmax_rowwise_jacobian(const Vector& x)
{
  assert(is_row_vector(x));
  auto D = x.cols();
  return identity<Vector>(D) - row_repeat(softmax_rowwise(x), D);
}

template <typename Matrix>
auto stable_log_softmax_rowwise(const Matrix& X)
{
  auto D = X.cols();
  auto Y = X - column_repeat(rows_max(X), D);
  return Y - column_repeat(log(rows_sum(exp(Y))), D);
}

template <typename Vector>
auto stable_log_softmax_rowwise_jacobian(const Vector& x)
{
  return log_softmax_rowwise_jacobian(x);
}

} // namespace nerva::eigen

#endif // NERVA_NEURAL_NETWORKS_SOFTMAX_FUNCTIONS_H
