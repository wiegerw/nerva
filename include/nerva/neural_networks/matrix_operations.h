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

//////////////////////////////////////////////////////////////////////////////

template <typename Matrix>
auto zeros(long m, long n = 1)
{
  return Matrix::Zeros(m, n);
}

template <typename Matrix>
auto ones(long m, long n = 1)
{
  return Matrix::Ones(m, n);
}

template <typename Matrix>
auto identity(long n)
{
  return Matrix::Identity(n, n);
}

template <class Matrix1, class Matrix2>
auto product(const Matrix1& X, const Matrix2& Y)
{
  return X * Y;
}

template <class Matrix1, class Matrix2>
auto hadamard(const Matrix1& X, const Matrix2& Y)
{
  return X.cwiseProduct(Y);
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

template <typename Matrix>
auto elements_sum(const Matrix& X)
{
  return X.sum();
}

template <typename Matrix>
auto column_repeat(const Matrix& x, long n)
{
  assert(is_column_vector(x));
  return x.rowwise().replicate(n);
}

template <typename Matrix>
auto row_repeat(const Matrix& x, long m)
{
  assert(is_row_vector(x));
  return x.colwise().replicate(m);
}

template <typename Matrix>
auto columns_sum(const Matrix& X)
{
  return X.colwise().sum();
}

template <typename Matrix>
auto rows_sum(const Matrix& X)
{
  return X.rowwise().sum();
}

// Returns a column vector with the maximum values of each row in X.
template <typename Matrix>
auto columns_max(const Matrix& x)
{
  return x.colwise().maxCoeff();
}

// Returns a row vector with the maximum values of each column in X.
template <typename Matrix>
auto rows_max(const Matrix& x)
{
  return x.rowwise().maxCoeff();
}

// Returns a row vector with the mean values of each column in X.
template <typename Matrix>
auto columns_mean(const Matrix& x)
{
  return x.colwise().mean();
}

// Returns a column vector with the mean values of each row in X.
template <typename Matrix>
auto rows_mean(const Matrix& x)
{
  return x.rowwise().mean();
}

template <typename Matrix, typename Function>
auto apply(Function f, const Matrix& X)
{
  return X.unaryExpr(f);
}

template <class Matrix>
auto exp(const Matrix& X)
{
  return X.array().exp().matrix();
}

template <class Matrix>
auto log(const Matrix& X)
{
  return X.array().log().matrix();
}

template <class Matrix>
auto inverse(const Matrix& X)
{
  return X.array().inverse().matrix();
}

template <class Matrix>
auto square(const Matrix& X)
{
  return X.array().square().matrix();
}

template <class Matrix>
auto sqrt(const Matrix& X)
{
  return X.array().sqrt().matrix();
}

template <typename Matrix>
auto power_minus_half(const Matrix& X)
{
  using Scalar = typename Matrix::Scalar;
  if constexpr (std::is_same<Scalar, float>::value)
  {
    return X.unaryExpr([](Scalar t) { return Scalar(1) / std::sqrt(t + Scalar(1e-15)); });
  }
  else
  {
    return X.unaryExpr([](Scalar t) { return Scalar(1) / std::sqrt(t + Scalar(1e-30)); });
  }
}

} // namespace nerva::eigen

#endif // NERVA_NEURAL_NETWORKS_MATRIX_OPERATIONS_H
