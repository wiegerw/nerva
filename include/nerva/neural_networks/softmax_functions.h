// Copyright: Wieger Wesselink 2022 - present
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/neural_networks/softmax_functions.h
/// \brief add your file description here.

#pragma once

#include "nerva/neural_networks/eigen.h"
#include <cassert>
#include "nerva/neural_networks/eigen.h"
#include "nerva/neural_networks/softmax_functions.h"
#include "fmt/format.h"
#include <cmath>
#include <ratio>

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
  return Diag(y).toDenseMatrix() - y * y.transpose();  // TODO: the .toDenseMatrix() should not be needed
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
  return Diag(y).toDenseMatrix() - y * y.transpose();  // TODO: the .toDenseMatrix() should not be needed
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
  return Diag(y.transpose()).toDenseMatrix() - y.transpose() * y;  // TODO: the .toDenseMatrix() should not be needed
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
  return Diag(y.transpose()).toDenseMatrix() - y.transpose() * y;  // TODO: the .toDenseMatrix() should not be needed
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


namespace nerva::colwise {

// N.B. Numerically unstable!
struct softmax
{
  [[nodiscard]] auto value(const eigen::vector& x) const -> eigen::vector
  {
    using eigen::exp;

    auto E = exp(x);
    return E / E.sum();
  }

  auto operator()(const eigen::matrix& X) const -> eigen::matrix
  {
    return nerva::eigen::softmax_colwise(X);
  }
};

struct stable_softmax
{
  [[nodiscard]] auto value(const eigen::vector& x) const -> eigen::vector
  {
    using eigen::exp;

    // use the log-sum-exp trick to make the computation robust, see also https://en.wikipedia.org/wiki/LogSumExp
    scalar const c = x.maxCoeff();
    auto E = exp((x.array() - c));
    return E / E.sum();
  }

  auto operator()(const eigen::matrix& X) const -> eigen::matrix
  {
    return nerva::eigen::stable_softmax_colwise(X);
  }
};

// N.B. Numerically unstable!
struct log_softmax
{
  [[nodiscard]] auto value(const eigen::vector& x) const -> eigen::vector
  {
    using eigen::row_repeat;
    using eigen::columns_sum;
    using eigen::exp;
    using eigen::log;

    auto N = x.size();
    auto e = log(columns_sum(exp(x)));
    return x - row_repeat(e, N);
  }

  auto operator()(const eigen::matrix& X) const -> eigen::matrix
  {
    return nerva::eigen::log_softmax_colwise(X);
  }
};

struct stable_log_softmax
{
  [[nodiscard]] auto value(const eigen::vector& x) const -> eigen::vector
  {
    using eigen::exp;

    auto c = x.array().maxCoeff();
    auto E = std::log(exp(x.array() - c).sum());
    return x.array() - c - E;
  }

  auto operator()(const eigen::matrix& X) const -> eigen::matrix
  {
    return nerva::eigen::stable_log_softmax_colwise(X);
  }
};

} // namespace nerva::colwise
